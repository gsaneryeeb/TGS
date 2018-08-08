# -*- coding: utf-8 -*-
"""
@auth Jason Zhang <gsangeryeee@gmail.com>
@brief: 

"""

import numpy as np
from hyperopt import hp

import config

# -------------------------------------- All ---------------------------------------------
param_space_dict = {
    # xgboost
    "reg_xgb_tree": param_space_reg_xgb_tree,
    "reg_xgb_tree_best_single_model": param_space_reg_xgb_tree_best_single_model,
    "reg_xgb_linear": param_space_reg_xgb_linear,
    "clf_xgb_tree": param_space_clf_xgb_tree,
    # sklearn
    "reg_skl_lasso": param_space_reg_skl_lasso,
    "reg_skl_ridge": param_space_reg_skl_ridge,
    "reg_skl_bayesian_ridge": param_space_reg_skl_bayesian_ridge,
    "reg_skl_random_ridge": param_space_reg_skl_random_ridge,
    "reg_skl_lsvr": param_space_reg_skl_lsvr,
    "reg_skl_svr": param_space_reg_skl_svr,
    "reg_skl_knn": param_space_reg_skl_knn,
    "reg_skl_etr": param_space_reg_skl_etr,
    "reg_skl_rf": param_space_reg_skl_rf,
    "reg_skl_gbm": param_space_reg_skl_gbm,
    "reg_skl_adaboost": param_space_reg_skl_adaboost,
    # keras
    "reg_keras_dnn": param_space_reg_keras_dnn,
    # rgf
    "reg_rgf": param_space_reg_rgf,
    # ensemble
    "reg_ensemble": param_space_reg_ensemble,
}

int_params = [
    "num_round", "n_estimators", "min_samples_split", "min_samples_leaf",
    "n_neighbors", "leaf_size", "seed", "random_state", "max_depth", "degree",
    "hidden_units", "hidden_layers", "batch_size", "nb_epoch", "dim", "iter",
    "factor", "iteration", "n_jobs", "max_leaf_forest", "num_iteration_opt",
    "num_tree_search", "min_pop", "opt_interval",
]
int_params = set(int_params)


class ModelParamSpace:
    def __init__(self, learner_name):
        s = "Wrong learner_name, " + \
            " see model_param_space.py for all available learners."
        assert learner_name in param_space_dict, s
        self.learner_name = learner_name

    def _build_space(self):
        return param_space_dict[self.learner_name]

    def _convert_int_param(self, param_dict):
        if isinstance(param_dict, dict):
            for k, v in param_dict.items():
                if k in int_params:
                    param_dict[k] = int(v)
                elif isinstance(v, list) or isinstance(v, tuple):
                    for i in range(len(v)):
                        self._convert_int_param(v[i])
                elif isinstance(v, dict):
                    self._convert_int_param(v)
        return param_dict
