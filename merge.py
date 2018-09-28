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
import pandas as pd

from sklearn.metrics import jaccard_similarity_score




# # TODO: 待完善
# threshold = best_threshold
# binary_prediction = (all_predictions_stacked > threshold).astype(int)
# all_masks = []
# for p_mask in list(binary_prediction):
#     p_mask = rle_encoding(p_mask)
#     all_masks.append(' '.join(map(str, p_mask)))

# # Make submission file
# submit = pd.DataFrame([test_file_list, all_masks]).T
# submit.columns = ['id', 'rel_mask']
# submit.to_csv('submit_unet_11_2018091901.csv.gz', compression = 'gzip', index = False)


# 合并 fold 中的结果
def merge(file_name):
    for fold in range(num_folds):
        df_fold_name = str(config.SUBMISSION_PATH /'tgsv1'/ str(fold)) + '.csv'
        print('df_fold_name',df_fold_name) 
        df = pd.read_csv(df_fold_name, sep=',')
        print(df[df['id'].isin([file_name.stem])])


if __name__ == '__main__':
    print("====Merge===")
    num_folds = 5
    model_name = 'unet_11'

    test_images = sorted(list((Path(model_name) / '0' / 'test').glob('*.png')))

    (config.SUBMISSION_PATH / 'tgsv1').mkdir(exist_ok=True, parents=True)

    # Parallel(n_jobs=16)(delayed(merge)(x) for x in test_images)
    for x in test_images:
        merge(x)

