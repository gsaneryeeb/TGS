# -*- coding: utf-8 -*-
"""
@auth Jason Zhang <gsangeryeee@gmail.com>
@brief: 

"""

from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed
import config


# 合并 fold 中的结果

def merge(file_name):
    result = np.zeros((num_folds, 101, 101))
    for fold in range(num_folds):
        img_path = file_name.parent.parent.parent / str(fold) / 'test' / (file_name.stem + '.png')
        img = cv2.imread(str(img_path), 0)
        result[fold] = img

    img = result.mean(axis=0).astype(np.uint8)
    # print("img:",img)
    print("File :",str(config.SUBMISSION_PATH / 'tgsv1' / (file_name.stem + '.png')))
    cv2.imwrite(str(config.SUBMISSION_PATH / 'tgsv1' / (file_name.stem + '.png')), img)


if __name__ == '__main__':
    print("====Merge===")
    num_folds = 5
    model_name = 'unet_11'

    test_images = sorted(list((Path(model_name) / '0' / 'test').glob('*.png')))

    (config.SUBMISSION_PATH / 'tgsv1').mkdir(exist_ok=True, parents=True)

    Parallel(n_jobs=16)(delayed(merge)(x) for x in test_images)

