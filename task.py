# -*- coding: utf-8 -*-
"""
@auth Jason Zhang <gsangeryeee@gmail.com>
@brief: 

"""

import argparse
import random
import os
from os.path import join
import numpy as np

import torch
import torch.backends.cudnn

import config

from utils import unet_utils



# -------------------------------main-------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Kaggle TGS Competition')
    parser.add_argument('-o', '--output_dir', default=None, help='output dir')
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=400, help='number of epochs to train')
    parser.add_argument('-lr', '--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate')
    parser.add_argument('-reset_lr', '--reset_lr', action='store_true',
                        help='should reset lr cycles? If not count epochs from 0')
    parser.add_argument('-opt', '--optimizer', default='sgd', choices=['sgd', 'adam', 'rmsprop'],
                        help='optimizer type')
    parser.add_argument('--decay_step', type=float, default=100, metavar='EPOCHS',
                        help='learning rate decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.5,
                        help='learning rate decay coeeficient')
    parser.add_argument('--cyclic_lr', type=int, default=None,
                        help='(int)Len of the cycle. If not None use cyclic lr with cycle_len) specified')
    parser.add_argument('--cyclic_duration', type=float, default=1.0,
                        help='multiplier of the duration of segments in the cycle')

    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='L2 regularizer weight')
    parser.add_argument('--seed', type=int, default=1993, help='random seed')
    parser.add_argument('--log_aggr', type=int, default=None, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-gacc', '--num_grad_acc_steps', type=int, default=1, metavar='N',
                        help='number of vatches to accumulate gradients')
    parser.add_argument('-imsize', '--image_size', type=int, default=1024, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-f', '--fold', type=int, default=0, metavar='N',
                        help='fold_id')
    parser.add_argument('-nf', '--n_folds', type=int, default=0, metavar='N',
                        help='number of folds')
    parser.add_argument('-fv', '--folds_version', type=int, default=1, choices=[1, 2],
                        help='version of folds (1) - random, (2) - stratified on mask area')
    parser.add_argument('-group', '--group', type=parse_group, default='all',
                        help='group id')
    parser.add_argument('-no_cudnn', '--no_cudnn', action='store_true',
                        help='dont use cudnn?')
    parser.add_argument('-aug', '--aug', type=int, default=None,
                        help='use augmentations?')
    parser.add_argument('-no_hq', '--no_hq', action='store_true',
                        help='do not use hq images?')
    parser.add_argument('-dbg', '--dbg', action='store_true',
                        help='is debug?')
    parser.add_argument('-is_log_dice', '--is_log_dice', action='store_true',
                        help='use -log(dice) in loss?')
    parser.add_argument('-no_weight_loss', '--no_weight_loss', action='store_true',
                        help='do not weight border in loss?')

    parser.add_argument('-suf', '--exp_suffix', default='', help='experiment suffix')
    parser.add_argument('-net', '--network', default='Unet')

    args = parser.parse_args()
    print("aug:", args.aug)
    # assert args.aug, 'Careful! No aug specified!'
    if args.log_aggr is None:
        args.log_aggr = 1
    print('log_aggr', args.log_aggr)

    # Set random seed
    random.seed(42)
    torch.manual_seed(args.seed)

    print('CudNN:', torch.backends.cudnn.version())
    print('Run on {} GPUs'.format(torch.cuda.device_count()))
    torch.backends.cudnn.benchmark = not args.no_cudnn  # Enable use of CudNN

    experiment = "{}_s{}_im{}_gacc{}{}{}{}_{}fold{}.{}"\
        .format(args.network, args.seed, args.image_size, args.num_grad_acc_steps,
                '_aug{]'.format(args.aug) if args.aug is not None else '',
                '_nohq' if args.no_hq else '',
                '_g{}'.format(args.group) if args.group != 'all' else '',
                'v2' if args.folds_version == 2 else '',
                args.fold, args.n_folds)

    if args.output_dir is None:
        ckpt_dir = join(config.MODELS_DIR, experiment + args.exp_suffix)
        if os.path.exists(join(ckpt_dir, 'checkpoint.pth.tar')):
            args.output_dir = ckpt_dir

    if args.output_dir is not None and os.path.exists(args.output_dir):
        ckpt_path = join(args.output_dir, 'checkpoint.pth.tar')
        if not os.path.isfile(ckpt_path):
            print("=> no checkpoint found at '{}'\nUsing model_best.pth.tar".format(ckpt_path))
            ckpt_path = join(args.output_dir, 'model_best.pth.tar')
        if os.path.isfile(ckpt_path):
            print("=> loading checkpoint '{}'".format(ckpt_path))
            checkpoint = torch.load(ckpt_path)
            if 'filters_sizes' in checkpoint:
                filters_sizes = checkpoint['filters_sizes']
            print("=> loaded checkpoint '{}' (epoch {})".format(ckpt_path, checkpoint['epoch']))
        else:
            raise IOError("=> no checkpoint found at '{}'".format(ckpt_path))
    else:
        checkpoint = None
        if args.network == 'UNet':
            filters_sizes = np.asarray([32, 64, 64, 128, 128, 256])
        else:
            raise ValueError('Unknown Net: {}'.format(args.network))

    if args.network in ['vgg11v1','vgg11v2']:
        pass
    elif args.network in ['vgg11av1','vgg11va2']:
        pass
    else:
        unet_class = getattr(unet_utils, args.network)
        model = torch.nn.DataParallel(
            unet_class(is_deconv=False, filters=filters_sizes)).cuda()
    print('  + Number of params: {}'.format(sum([p.data.nelment() for p in model.parameters()])))

    rescale_size = (args.image_size, args.image_size)

    # Load train data



if __name__ == '__main__':
    main()







































