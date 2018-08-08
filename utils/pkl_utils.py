# -*- coding: utf-8 -*-
"""
@auth Jason Zhang <gsangeryeee@gmail.com>
@brief: 

"""

import pickle


def _save(fname, data, protocol=3):
    with open(fname, "wb") as f:
        pickle.dump(data, f, protocol)

def _load(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)
