# -*- coding: utf-8 -*-
"""
@auth Jason Zhang <gsangeryeee@gmail.com>
@brief: 

"""

import platform

import numpy as np


import json
from pathlib import Path


# config = json.loads(open(str(Path('__file__').absolute().parent.parent / 'code' / 'config.json')).read())
config = json.loads(open('config.json').read())


DATA_ROOT = Path(config['input_data_dir']).expanduser()

MODELS_DIR = Path(config['models_dir']).expanduser()

SUBMISSION_PATH = Path(config['submissions_dir']).expanduser()

def add_args(parser):
    arg = parser.add_argument
    arg('--root', default='models/unet_11', help='checkpoint root')
    arg('--batch-size', type=int, default=24)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=8)
    arg('--clean', action='store_true')
    arg('--epoch-size', type=int)
