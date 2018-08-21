# -*- coding: utf-8 -*-
"""
@auth Jason Zhang <gsangeryeee@gmail.com>
@brief: 

"""

import json
from pathlib import Path


config = json.loads(open(str(Path('__file__').absolute().parent.parent / 'code' / 'config.json')).read())

print(config)

DATA_ROOT = Path(config['input_data_dir']).expanduser()

print(DATA_ROOT)
