# -*- coding: utf-8 -*-
"""
@auth Jason Zhang <gsangeryeee@gmail.com>
@brief: 

"""

from pathlib import Path
import shutil

import pandas as pd
from tqdm import tqdm

import config


if __name__ == '__main__':
    global_data_path = config.DATA_ROOT

    local_data_path = Path('.').absolute()

    local_data_path.mkdir(exist_ok=True)

    train_path = global_data_path / 'train' / 'images'

    mask_path = global_data_path / 'train' / 'masks'

    train_file_list = train_path.glob('*')

    folds = pd.read_csv('train_folds.csv')

    num_folds = folds['fold'].nunique()

    print("num_folds", num_folds)

    for fold in range(num_folds):

        (local_data_path / str(fold) / 'train' / 'images').mkdir(exist_ok=True, parents=True)
        (local_data_path / str(fold) / 'train' / 'masks').mkdir(exist_ok=True, parents=True)

        (local_data_path / str(fold) / 'val' / 'images').mkdir(exist_ok=True, parents=True)
        (local_data_path / str(fold) / 'val' / 'masks').mkdir(exist_ok=True, parents=True)


    for i in tqdm(folds.index):
        image_id = folds.loc[i, 'image']
        fold = folds.loc[i, 'fold']

        # Copy all images (with fold number) into /fold/val/images and /fold/val/masks
        old_image_path = train_path / (image_id + '.png')
        new_image_path = local_data_path / str(fold) / 'val' / 'images' / (image_id + '.png')
        shutil.copy(str(old_image_path), str(new_image_path))

        old_mask_path = mask_path / (image_id + '.png')
        new_mask_path = local_data_path / str(fold) / 'val' / 'masks' / (image_id + '.png')

        shutil.copy(str(old_mask_path), str(new_mask_path))

        for t_fold in range(num_folds):
            if t_fold == fold:
                continue

            old_image_path = train_path / (image_id + '.png')
            new_image_path = local_data_path / str(t_fold) / 'train' / 'images' / (image_id + '.png')
            shutil.copy(str(old_image_path), str(new_image_path))

            old_mask_path = mask_path / (image_id + '.png')
            new_mask_path = local_data_path / str(t_fold) / 'train' / 'masks' / (image_id + '.png')

            shutil.copy(str(old_mask_path), str(new_mask_path))



















