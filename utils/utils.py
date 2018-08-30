# -*- coding: utf-8 -*-
"""
@auth Jason Zhang <gsangeryeee@gmail.com>
@brief: 

"""


import config
import torch



from torchvision.transforms import ToTensor, Normalize, Compose



# https://stackoverflow.com/questions/50002543/transforms-compose-meaning-pytorch
# 对图像进行归一化处理。
img_transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])






