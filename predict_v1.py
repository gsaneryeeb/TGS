# -*- coding: utf-8 -*-
"""
@auth Jason Zhang <gsangeryeee@gmail.com>
@brief: 

"""
import argparse
import utils
import cv2
from pathlib import Path

import task_v1
import config 
import numpy as np

from utils.unet_vgg_utils import UNet11
from utils import utils

import torch
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import pandas as pd



class PredictionDatasetPure:
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        image = task_v1.load_image(path)
        return utils.img_transform(image), path.stem


def get_model(model_name):
    model = UNet11()
    model = nn.DataParallel(model, device_ids=[0]).cuda()

    state = torch.load(
        str(config.MODELS_DIR / model_name / 'best-model_{fold}.pt'.format(fold=fold, model_name=model_name)))
    model.load_state_dict(state['model'])
    model.eval() # 评估模式

    return model

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def predict(model, from_paths, batch_size: int, to_path):
    all_predictions = []
    loader = DataLoader(
        dataset=PredictionDatasetPure(from_paths),
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=True
    )
    height, width = 101, 101

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

    for batch_num, (inputs, stems) in enumerate(tqdm.tqdm(loader, desc='Predict')):
        # print("predict_v1.py inputs size:", inputs.size()) # input size(2,3,128,128)
        inputs = task_v1.variable(inputs, volatile=True)
        outputs = model(inputs)  # shape (2, 1, 128, 128)
        # print("predict_v1.py outputs size:", outputs.size())
        # print("predict_v1.py outputs:", outputs)
        all_predictions.append(outputs.cpu().detach().numpy())
        mask = (outputs.data.cpu().numpy() * 255).astype(np.uint8) # shape(2, 1, 128, 128) 恢复为原始数值
        for i, image_name in enumerate(stems):
            

            # print('mask shape:', mask.shape)
            # print('mask :', mask)
            # print('mask image shape:', mask[i, 0, :, :].shape)
            # print('mask image shape:', mask[i, 0, :, :])
            # print('mask image shape:', mask[i, 0, :, :][y_min_pad:128 - y_max_pad,x_min_pad:128-x_max_pad].shape) 
            cv2.imwrite(str(to_path / (stems[i] + '.png')), mask[i, 0, :, :][y_min_pad:128 - y_max_pad,x_min_pad:128-x_max_pad])
    # print("all_prediction:", all_predictions)
    all_predictions_stacked = np.vstack(all_predictions)[:, 0, :, :]  # 降维
    all_predictions_stacked = all_predictions_stacked[:, y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad] # Padding 128x128 --> 101x101
    # print("all_preidction shape:", all_predictions_stacked.shape)
    return all_predictions_stacked

if __name__ == '__main__':
    local_data_path = Path('.').absolute()
    local_data_path.mkdir(exist_ok=True)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--fold', type=int)
    
    utils.add_args(parser)
    args = parser.parse_args()

    model_path = config.MODELS_DIR
    data_path = config.DATA_ROOT

    model_name = 'unet_11'

    # Create prediction folder
    pred_path = Path(model_name)
    pred_path.mkdir(exist_ok=True, parents=True)

    fold_path = pred_path / str(args.fold)

    # Create validatition folder
    val_path = fold_path / 'val'
    val_path.mkdir(exist_ok=True, parents=True)
    
    # Create test folder
    test_path = fold_path / 'test'
    test_path.mkdir(exist_ok=True, parents=True)

    fold = args.fold
    batch_size = 2

    model = get_model(model_name)
    
    val_masks_path = Path('.').absolute() / str(args.fold) / 'val' / 'masks'
    print('fold mask path=', val_masks_path)

    val_images = sorted(list((Path(str(args.fold)) / 'val' / 'images').glob('*.png')))
    num_val = len(val_images)
    val_pred_masks = predict(model, val_images, batch_size, val_path)

    # TODO 获取每个fold的mask文件计算
    # val_masks = task_v1.load_image(val_masks_path, mask=True)

    test_images = sorted(list((data_path / 'test'/ 'images').glob('*.png')))
    
    test_file_list = [f.stem for f in test_images] # 只获取文件名
    num_test = len(test_images)
    pred_maks = predict(model, test_images, batch_size, test_path)

    all_fold_masks = []
    for p_mask in list(pred_maks):
        all_fold_masks.append(' '.join(map(str, p_mask)))


    submit = pd.DataFrame([test_file_list, all_fold_masks])
    submit.columns = ['id','mask']
    submit.to_csv(str(config.SUBMISSION_PATH / 'tgsv1' / str(fold)+'.csv'), index=False)
    #sub
    # threshold = 0.5
    # binary_prediction = (pred_maks > threshold).astype(int)
    
    # all_fold_masks = []
    # for p_mask in list(binary_prediction):
    #     p_mask = rle_encoding(p_mask)
    #     all_fold_masks.append(' '.join(map(str, p_mask)))
    # submit = pd.DataFrame([test_file_list, all_fold_masks]).T
    # submit.columns = ['id','rle_mask']
    # submit_file_name = 'submit_'+str(fold)+'.csv'
    # submit.to_csv(str(config.SUBMISSION_PATH / 'tgsv1' / str(fold)+'.csv'), index=False)
    # submit = pd.DataFrame([test_images,pred_maks]).T
    # submit.columns = ['id', 'mask']
    # submit_file_name = 'submit_'+str(fold)+'.csv'
