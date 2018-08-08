# -*- coding: utf-8 -*-
"""
@auth Jason Zhang <gsangeryeee@gmail.com>
@brief: 

"""

import platform

import numpy as np


# ------------------- Overall -------------------
TASK = "all"
# # for testing data processing and feature generation
# TASK = "sample"
SAMPLE_SIZE = 1000


# ------------------- PATH -------------------
ROOT_DIR = "../.."

DATA_DIR = "%s/Data" % ROOT_DIR

FEAT_DIR = "%s/Feat"%ROOT_DIR
FEAT_FILE_SUFFIX = ".pkl"
FEAT_CONF_DIR = "./conf"

OUTPUT_DIR = "%s/Output" % ROOT_DIR
SUBM_DIR = "%s/Subm" % ROOT_DIR

LOG_DIR = "%s/Log" % ROOT_DIR
FIG_DIR = "%s/Fig"%ROOT_DIR

# ------------------- DATA -------------------
# provided data
TRAIN_DATA = "%s/train.csv" % DATA_DIR
TEST_DATA = "%s/test.csv" % DATA_DIR



