"""
XGBoost Parameters Configuration for Student Performance Prediction

This module contains the optimized XGBoost hyperparameters for each question in the
student performance prediction pipeline. These parameters were carefully tuned through
extensive hyperparameter optimization and cross-validation to achieve the best possible
performance for each individual question.

The parameters are organized by question number and include both the XGBoost model
parameters and the optimal threshold (umbral) for binary classification.

Key Components:
- xgb_params_by_question: Dictionary mapping question numbers to optimized parameters
- params: XGBoost model hyperparameters for each question
- umbral: Optimal classification threshold for each question
- best_its: Best iteration numbers for early stopping (when applicable)

Parameter Optimization Strategy:
1. Question-specific hyperparameter tuning using cross-validation
2. Different parameter sets for different question types
3. Optimized thresholds for each question's class distribution
4. Early stopping parameters to prevent overfitting
5. Memory and performance optimizations (tree_method: "hist")

Author: Competition Team
Competition: Predict Student Performance from Game Play
Position: 35/2051 (Silver Medal)
"""

xgb_params_by_question = {
    1: {
        "params": {
            "best_its": (800, 800),
            "max_depth": 6,
            "scale_pos_weight": 0.39999999999999997,
            "learning_rate": 0.022,
            "tree_method": "hist",
            "n_jobs": 16,
            "random_state": 32,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "grow_policy": "lossguide",
        },
        "umbral": 0.5000000000000001,
    },
    4: {
        "params": {
            "best_its": [(539, 691), (497, 582), (891, 867), (599, 607)],
            "max_depth": 4,
            "scale_pos_weight": 0.7,
            "learning_rate": 0.027,
            "tree_method": "hist",
            "n_jobs": 16,
            "random_state": 32,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "grow_policy": "lossguide",
        },
        "umbral": 0.5600000000000002,
    },
    5: {
        "params": {
            "best_its": [(564, 564), (468, 478), (306, 384), (341, 478)],
            "max_depth": 5,
            "grow_policy": "lossguide",
            "scale_pos_weight": 0.8999999999999999,
            "learning_rate": 0.022,
            "colsample_bytree": 0.4,
            "subsample": 0.6,
            "tree_method": "hist",
            "n_jobs": 16,
            "random_state": 32,
        },
        "umbral": 0.6000000000000003,
    },
    6: {
        "params": {
            "best_its": (650, 650),
            "max_depth": 5,
            "scale_pos_weight": 0.6,
            "learning_rate": 0.02,
            "tree_method": "hist",
            "n_jobs": 16,
            "random_state": 32,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "grow_policy": "lossguide",
        },
        "umbral": 0.5500000000000002,
    },
    7: {
        "params": {
            "best_its": (800, 800),
            "max_depth": 4,
            "scale_pos_weight": 0.6,
            "learning_rate": 0.02,
            "tree_method": "hist",
            "n_jobs": 16,
            "random_state": 32,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "grow_policy": "lossguide",
        },
        "umbral": 0.5200000000000001,
    },
    8: {
        "params": {
            "best_its": (350, 350),
            "max_depth": 6,
            "scale_pos_weight": 1.0999999999999999,
            "learning_rate": 0.015,
            "tree_method": "hist",
            "n_jobs": 16,
            "random_state": 32,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "grow_policy": "lossguide",
        },
        "umbral": 0.6600000000000003,
    },
    9: {
        "params": {
            "best_its": (699, 763),
            "max_depth": 5,
            "scale_pos_weight": 0.5,
            "learning_rate": 0.02,
            "tree_method": "hist",
            "n_jobs": 16,
            "random_state": 32,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "grow_policy": "lossguide",
        },
        "umbral": 0.4900000000000001,
    },
    10: {
        "params": {
            "best_its": [(397, 377), (374, 381), (412, 329), (253, 336)],
            "max_depth": 6,
            "scale_pos_weight": 0.7999999999999999,
            "learning_rate": 0.02,
            "tree_method": "hist",
            "n_jobs": 16,
            "random_state": 32,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "grow_policy": "lossguide",
        },
        "umbral": 0.5700000000000002,
    },
    11: {
        "params": {
            "best_its": (450, 450),
            "max_depth": 6,
            "scale_pos_weight": 0.7999999999999999,
            "learning_rate": 0.022,
            "tree_method": "hist",
            "n_jobs": 16,
            "random_state": 32,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "grow_policy": "lossguide",
        },
        "umbral": 0.5900000000000002,
    },
    14: {
        "params": {
            "best_its": (450, 450),
            "max_depth": 5,
            "scale_pos_weight": 0.9999999999999999,
            "learning_rate": 0.02,
            "tree_method": "hist",
            "n_jobs": 16,
            "random_state": 32,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "grow_policy": "lossguide",
        },
        "umbral": 0.6400000000000002,
    },
    15: {
        "params": {
            "best_its": (700, 700),
            "max_depth": 6,
            "scale_pos_weight": 0.5,
            "learning_rate": 0.02,
            "tree_method": "hist",
            "n_jobs": 16,
            "random_state": 32,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "grow_policy": "lossguide",
        },
        "umbral": 0.5100000000000001,
    },
    16: {
        "params": {
            "best_its": (500, 500),
            "max_depth": 4,
            "scale_pos_weight": 0.6,
            "learning_rate": 0.02,
            "tree_method": "hist",
            "n_jobs": 16,
            "random_state": 32,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "grow_policy": "lossguide",
        },
        "umbral": 0.5100000000000001,
    },
    17: {
        "params": {
            "best_its": (195, 225),
            "max_depth": 5,
            "scale_pos_weight": 0.8999999999999999,
            "learning_rate": 0.02,
            "tree_method": "hist",
            "n_jobs": 16,
            "random_state": 32,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "grow_policy": "lossguide",
        },
        "umbral": 0.5900000000000002,
    },
    2: {
        "params": {
            "booster": "gbtree",
            "tree_method": "hist",
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "learning_rate": 0.02,
            "alpha": 8,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.5,
            "seed": 42,
            "n_estimators": 448,
        },
        "umbral": 0.625,
        "best_iterations": None,
    },
    3: {
        "params": {
            "n_estimators": 296,
            "max_depth": 5,
            "grow_policy": "lossguide",
            "scale_pos_weight": 1.0999999999999999,
            "learning_rate": 0.02,
            "colsample_bytree": 0.7,
            "subsample": 0.8,
            "tree_method": "hist",
            "n_jobs": 16,
            "random_state": 32,
            "eval_metric": "logloss",
            "best_its": (292, 296),
        },
        "umbral": 0.6600000000000004,
        "best_iterations": None,
    },
    12: {
        "params": {
            "n_estimators": 236,
            "max_depth": 5,
            "grow_policy": "lossguide",
            "scale_pos_weight": 1.0999999999999999,
            "learning_rate": 0.02,
            "colsample_bytree": 0.7,
            "subsample": 0.8,
            "tree_method": "hist",
            "n_jobs": 16,
            "random_state": 32,
            "eval_metric": "logloss",
            "best_its": (197, 236),
        },
        "umbral": 0.6800000000000004,
        "best_iterations": None,
    },
    13: {
        "params": {
            "n_estimators": 255,
            "max_depth": 5,
            "grow_policy": "lossguide",
            "scale_pos_weight": 1.0999999999999999,
            "learning_rate": 0.02,
            "colsample_bytree": 0.7,
            "subsample": 0.8,
            "tree_method": "hist",
            "n_jobs": 16,
            "random_state": 32,
            "eval_metric": "logloss",
            "best_its": (276, 255),
        },
        "umbral": 0.6800000000000004,
        "best_iterations": None,
    },
}
