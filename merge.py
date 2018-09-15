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


def merge_test(file_name):
    result = np.zeros((num_folds, 128, 128))
    for fold in range(num_folds):
        img_path = file_name.parent.parent.parent / str(fold) / 'test' / (file_name.stem + '.png')
        img = cv2.imread(str(img_path), 0)
        result[fold] = img

    img = result.mean(axis=0).astype(np.uint8)
    print("img:",img)
    print("File Name:",file_name.stem)
    cv2.imwrite(str(config.SUBMISSION_PATH / 'tgsv1' / (file_name.stem + '.png')), img)


if __name__ == '__main__':
    print("====Merge===")
    num_folds = 5
    model_name = 'unet_11'

    test_images = sorted(list((Path(model_name) / '0' / 'test').glob('*.png')))

    (config.SUBMISSION_PATH / model_name / 'test_averaged').mkdir(exist_ok=True, parents=True)

    Parallel(n_jobs=16)(delayed(merge_test)(x) for x in test_images)

