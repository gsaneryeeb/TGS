# -*- coding: utf-8 -*-
"""
@auth Jason Zhang <gsangeryeee@gmail.com>
@brief: 
@2018-09-11
test1v
"""
# TODO 1: Check augment function
# TODO 2:

import json
import argparse
import cv2
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

import random
import tqdm
import shutil

from datetime import datetime

from itertools import islice
from config import config

import torch

from torch import nn
from torch.optim import Adam
from torch.autograd import Variable

from utils.unet_vgg_utils import Loss, UNet11
from utils import utils
from torch.utils.data import DataLoader, Dataset

import config


cuda_is_available = torch.cuda.is_available()


class TGSDataset(Dataset):
    def __init__(self, root: Path, to_augment=False, is_test=False):
        self.is_test = is_test
        self.to_augment = to_augment
        self.image_paths = sorted(root.joinpath('images').glob('*.png'))
        self.mask_paths = sorted(root.joinpath('mask').glob('*.png'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        print('Image length:', len(self.image_paths))
        print('Mask length:', len(self.mask_paths))
        image = load_image(self.image_paths[index])

        if self.is_test:

            return (image,) # 测试数据取值时需要用image[0]

        else:
            mask = load_image(self.mask_paths[index], mask=True)
            if self.to_augment:
                image, mask = augment(image, mask)

            return utils.img_transform(image), torch.from_numpy(mask).permute([2, 0, 1])


def load_image(path: Path, mask=False):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (newtwork requirement)

    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """
    img = cv2.imread(str(path))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB 为后面变化使用。cv2 读取图片的默认通道是BGR

    height, width, _ = img.shape

    # Padding in needed for UNet models because they need image size to be divisible by 32
    if height % 32 == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = 32 - height % 32
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad

    if width % 32 == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = 32 - width % 32
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    # TODO: 此种方式在边界处会影响预测效果。在图像增强可以考虑，旋转后在padding，或者缩减的方式，按照32倍数窗口进行切割并预测。
    # Various border types, image boundaries aredenoted with '|'
    # BORDER_REPLICATE: aaaaaa | abcdefgh | hhhhhhh
    # BORDER_REFLECT: fedcba | abcdefgh | hgfedcb
    # BORDER_REFLECT_101: gfedcb | abcdefgh | gfedcba
    # BORDER_WRAP: cdefgh | abcdefgh | abcdefg
    # BORDER_CONSTANT: iiiiii | abcdefgh | iiiiiii with some specified 'i'
    # 
    # 将图片维度 padding 为 3x128x128
    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

    # Version 3
    if mask:
        # img = (np.asarray(img) > 0).astype(np.float32)
        img = img[:, :, 0:1] // 255 # mask 图片只取一个通道值，因为三个通道值一样都为255（白色）
    else:
        img = np.asarray(img)
    return img.astype(np.float32)


    # # Version 2
    # if mask:
    #     img = img[:, :, 0:1]
    #
    # return torch.from_numpy(img).float().permute([2, 0, 1])  # 可能与model的网络结构有关，不转换会报错。

    # Version 1
    # if mask:
    #     # Convert mask to 0 and 1 format
    #     # mask 图片处理
    #     # 1. 取 mask 图片的 B 通道第一层，因为B通道所有值都是255(白色),[:,:,0:1] 全部的R和G和B的第一层
    #     # 2. // 255,取整除法操作，因为白色地方数值为255，取整除法后为1。
    #     # 3. 转换后所有白色部分，变为1（白色），其他为0（黑色）
    #     img = img[:, :, 0:1] // 255
    #     return torch.from_numpy(img).float().permute([2, 0, 1])
    #
    # else:
    #     img = img / 255.0  # 将图像值控制在0-1之间，方便计算。
    #     return torch.from_numpy(img).float().permute([2, 0, 1])
    #     # from_numpy : creates a Tensor（张量） from a numpy.ndarray
    #     # permute 变换维度顺序 原顺序为[0,1,2],变为[2,0,1]


def validation(model: nn.Module, criterion, valid_loader) -> Dict[str, float]:
    model.eval()  # 必备，将模型设置为评估模式
    losses = []
    dice = []

    for inputs, targets in valid_loader:
        inputs = variable(inputs, volatile=True)
        targets = variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.data[0])
        dice += [get_dice(targets, (outputs > 0.5).float()).data[0]]

    valid_loss = np.mean(losses)  # type: float

    valid_dice = np.mean(dice)

    print('Valid loss: {:.5f}, dice: {:.5f}'.format(valid_loss, valid_dice))
    metrics = {'valid_loss': valid_loss, 'dice_loss': valid_dice}
    return metrics


def get_dice(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1) + epsilon

    return 2 * (intersection / union).mean()


def augment(image, mask):
    if np.random.random() < 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)

    if np.random.random() < 0.5:
        if np.random.random() < 0.5:
            image = random_hue_saturation_value(image,
                                                hue_shift_limit=(-50, 50),
                                                sat_shift_limit=(-5, 5),
                                                val_shift_limit=(-15, 15))
        else:
            image = grayscale_aug(image, mask)

    return image.copy(), mask.copy()


def random_hue_saturation_value(image,
                                hue_shift_limit=(-100, 100),
                                sat_shift_limit=(-255, 255),
                                val_shift_limit=(-255, 255)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image)
    hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
    h = cv2.add(h, hue_shift)
    sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
    s = cv2.add(s, sat_shift)
    val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
    v = cv2.add(v, val_shift)
    image = cv2.merge((h, s, v))
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def grayscale_aug(image, mask):

    image_pixels = (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * image).astype(np.uint8)

    gray_image = cv2.cvtColor(image_pixels, cv2.COLOR_RGB2GRAY)

    rgb_gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

    rgb_img = image.copy()
    rgb_img[rgb_gray_image > 0] = rgb_gray_image[rgb_gray_image > 0]
    return rgb_img





# cyclic learning rate
def cyclic_lr(epoch, init_lr=1e-4, num_epochs_per_cycle=5, cycle_epochs_decay=2, lr_decay_factor=0.5):
    epoch_in_cycle = epoch % num_epochs_per_cycle
    lr = init_lr * (lr_decay_factor ** (epoch_in_cycle // cycle_epochs_decay)) # // 整数除，只保留整数部分
    return lr


def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(Variable(x, volatile=volatile))


def cuda(x):
    return x.cuda(async=True) if cuda_is_available else x


def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True, cls=MyEncoder))
    log.write('\n')
    log.flush()


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def train(args, model: nn.Module, criterion, *, train_loader, valid_loader,
          validation, init_optimizer, fold=None, save_predictions=None, n_epochs=None):

    n_epochs = n_epochs or args.n_epochs
    root = Path(args.root)
    print("train root=",root)
    model_path = root / 'model_{fold}.pt'.format(fold=fold)
    best_model_path = root / 'best-model_{fold}.pt'.format(fold=fold)
    print('model path =', model_path)
    print('best_model_path =', best_model_path)
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        best_valid_loss = state['best_valid_loss']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0
        best_valid_loss = float('inf')

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'best_valid_loss': best_valid_loss
    }, str(model_path))

    report_each = 10
    save_prediction_each = report_each * 20
    log = root.joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')
    valid_losses = []

    for epoch in range(epoch, n_epochs + 1):
        lr = cyclic_lr(epoch)

        optimizer = init_optimizer(lr)

        model.train()  # 必须，将模型设置为训练模式
        random.seed()
        tq = tqdm.tqdm(total=(args.epoch_size or
                              len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        if args.epoch_size:
            tl = islice(tl, args.epoch_size // args.batch_size)
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):  # 从数据加载器迭代一个batch的数据
                inputs, targets = variable(inputs), variable(targets)   # 使用GPU存储数据
                print("task_v1.py L326 inputs size", inputs.size())
                print("task_v1.py L326 targets size", targets.size())
                outputs = model(inputs)  # 喂入数据并前向传播获取输出
                print("task_v1.py L326 outputs size", outputs.size())
                # TODO Error Check inputs, outputs , targets shapes
                """ Kaggle 成功运行的维度
                image size torch.Size([30, 3, 128, 128])
                y_pred size torch.Size([30, 1, 128, 128])
                mask size torch.Size([30, 1, 128, 128])
                """
                # using augment
                # inputs = torch.Size([4, 3, 128, 128])
                # outputs = torch.Size([4, 1, 128, 128])
                # targets = torch.Size([4, 1, 128, 128, 3])
                # Without augment
                # inputs size =  torch.Size([4, 3, 128, 128])
                # outputs size =  torch.Size([4, 1, 128, 128])
                # targets size =  torch.Size([4, 1, 128, 128])
                loss = criterion(outputs, targets)  # 调用损失函数计算损失
                print("loss =", loss)
                optimizer.zero_grad()  # 清除所有优化的梯度
                batch_size = inputs.size(0)
                step += 1
                tq.update(batch_size)
                losses.append(loss.data[0])
                mean_loss = np.mean(losses[-report_each:])

                tq.set_postfix(loss='{:.5f}'.format(mean_loss))

                (batch_size * loss).backward()  # 反向传播
                optimizer.step()  # 更新参数

                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
                    if save_predictions and i % save_prediction_each == 0:
                        p_i = (i // save_prediction_each) % 5
                        save_predictions(root, p_i, inputs, targets, outputs)
            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1) # Save model
            valid_metrics = validation(model, criterion, valid_loader)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                shutil.copy(str(model_path), str(best_model_path))
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--dice-weight', type=float)
    arg('--nll-weights', action='store_true')
    arg('--device-ids', type=str, help='For example 0,1 to run on two GPUs')
    arg('--fold', type=int, help='fold', default=0)
    arg('--size', type=str, default='101x101', help='Input size, for example 288x384. Must be multiples of 32')
    utils.add_args(parser)
    args = parser.parse_args()

    model_name = 'unet_11'

    args.root = str(config.MODELS_DIR / model_name)

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    model = UNet11()

    device_ids = list(map(int, args.device_ids.split(',')))

    model = nn.DataParallel(model, device_ids=device_ids).cuda() # Using GPU

    loss = Loss()

    def make_loader(ds_root: Path, is_test=False, to_augment=False, shuffle=False):
        return DataLoader(
            dataset=TGSDataset(ds_root, is_test=is_test, to_augment=to_augment),
            shuffle=shuffle,
            num_workers=args.workers,
            batch_size=args.batch_size,
            pin_memory=True
        )

    # train_df = pd.read_csv(os.path.join(config.DATA_ROOT, 'train.csv'))

    # train_root = os.path.join(config.DATA_ROOT, 'train')
    # file_list = list(train_df['id'].values)
    # file_list_valid = file_list[::10]
    # file_list_train = [f for f in file_list if f not in file_list_valid]

    train_root = Path('.').absolute() / str(args.fold) / 'train'
    valid_root = Path('.').absolute() / str(args.fold) / 'val' 

    valid_loader = make_loader(valid_root)
    train_loader = make_loader(train_root, is_test=False, to_augment=False, shuffle=True)

    train(
        init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
        args=args,
        model=model,
        criterion=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=validation,
        fold=args.fold
    )


if __name__ == '__main__':
    main()

