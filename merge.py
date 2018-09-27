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

from sklearn.metrics import jaccard_similarity_score

def threshold(val_predictions, val_masks):
    metric_by_threshold = []
    for threshold in np.linspace(0, 1, 11): #(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
        val_binary_prediction = (val_predictions > threshold).astype(int)
        iou_values = []
        for y_mask, p_mask in zip(val_masks, val_binary_prediction):
            iou = jaccard_similarity_score(y_mask.flatten(), p_mask.flatten())
            iou_values.append(iou)
        iou_values = np.array(iou_values)
        accuracies = [
            np.mean(iou_values > iou_threshold)
            for iou_threshold in np.linspace(0.5, 0.95, 10)
        ]
        print('Threshold: %.2f, Metric: %.5f' % (threshold, np.mean(accuracies)))
        metric_by_threshold.append((np.mean(accuracies), threshold))
    best_metric, best_threshold = max(metric_by_threshold)
    return best_metric, best_threshold 

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

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
    result = np.zeros((num_folds, 101, 101))
    for fold in range(num_folds):
        # TODO: 1. 每个 fold 作为一组：比较 val 与 val mask 计算 该 fold 内 threshold。
        val_pred_images = sorted(list((file_name.parent.parent.parent / str(fold) / 'val').glob('*.png')))
        val_mask_imges = sorted(list((Path('.').absolute() / str(fold) / 'val' / 'images').glob('*.png')))
        pred_images = sorted(list((file_name.parent.parent.parent / str(fold) / 'test').glob('*.png')))

        # print('val_pred_images:', len(val_pred_images))
        # print('val_mask_imges:', len(val_mask_imges))
        # print('pred_images:', len(pred_images))

        # TODO: 2. 根据 threshold 修改该 fold 内的 pred 值
        # TODO: 3. 5 个 threshold 后的 pred 取 mean
        img_path = file_name.parent.parent.parent / str(fold) / 'test' / (file_name.stem + '.png')
        # print('im_path:', img_path)
        img = cv2.imread(str(img_path), 0)
        result[fold] = img

    # TODO: 4. mean 后输出submission
    img = result.mean(axis=0).astype(np.uint8)
    # print("img:", img)
    # print("File :",str(config.SUBMISSION_PATH / 'tgsv1' / (file_name.stem + '.png')))
    cv2.imwrite(str(config.SUBMISSION_PATH / 'tgsv1' / (file_name.stem + '.png')), img)


if __name__ == '__main__':
    print("====Merge===")
    num_folds = 5
    model_name = 'unet_11'

    test_images = sorted(list((Path(model_name) / '0' / 'test').glob('*.png')))

    (config.SUBMISSION_PATH / 'tgsv1').mkdir(exist_ok=True, parents=True)

    Parallel(n_jobs=16)(delayed(merge)(x) for x in test_images)

