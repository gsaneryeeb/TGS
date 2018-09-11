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


