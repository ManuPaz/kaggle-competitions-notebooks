"""
Configuration parameters and constants for Parkinson's Disease Progression Prediction.

This module contains all the configuration parameters, model hyperparameters,
and constants used throughout the Parkinson's disease prediction pipeline.

Author: [Your Name]
Date: [Current Date]
"""

import sklearn

# =============================================================================
# DATA PATHS AND DIRECTORIES
# =============================================================================

# Input directory containing the competition data files
INPUT_DIR = "../data/amp_parkinsons_disease_progression_prediction"

# Flag to control whether to run Kaggle inference or just training
KAGGLE_INFERENCE = False

# =============================================================================
# FEATURE ENGINEERING PARAMETERS
# =============================================================================

# Number of peptide and protein candidates to select for feature engineering
# This determines the number of features used in the model
NUM_CANDIDATES = 5

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Primary model to use for training and prediction
# Options: "xgboost", "linear", "svr"
MODEL_USE = "svr"

# =============================================================================
# XGBOOST MODEL PARAMETERS
# =============================================================================

XGBOOST_PARAMS = {
    "n_estimators": 10,  # Number of boosting rounds
    "max_depth": 4,  # Maximum depth of individual trees
    "learning_rate": 0.3,  # Learning rate (eta) for boosting
    "colsample_bytree": 1,  # Fraction of features used per tree
    "subsample": 1,  # Fraction of samples used per tree
    "tree_method": "hist",  # Tree construction algorithm
    "n_jobs": 12,  # Number of parallel threads
    "gamma": 0,  # Minimum loss reduction for split
    "reg_lambda": 0,  # L2 regularization term
}

# =============================================================================
# SVR MODEL PARAMETERS
# =============================================================================

SVR_PARAMS = {"kernel": "rbf", "degree": 2, "C": 2, "epsilon": 2}

# =============================================================================
# MODEL REGISTRY
# =============================================================================

# Dictionary mapping model names to their classes and parameters
# This allows easy switching between different model types
MODELS_DICT = {
    "xgboost": {"model": "xgboost.XGBRegressor", "params": XGBOOST_PARAMS},
    "linear": {"model": sklearn.linear_model.LinearRegression, "params": {"fit_intercept": True}},
    "svr": {"model": sklearn.svm.SVR, "params": SVR_PARAMS},
}

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

# Test size for train-test split (percentage of data for testing)
TEST_SIZE = 0.15

# Random seed for reproducibility
RANDOM_STATE = 42

# =============================================================================
# UPDRS SCORE PARTS
# =============================================================================

# UPDRS parts to predict (1-4)
UPDRS_PARTS = [1, 2, 3, 4]

# Time horizons for prediction (months ahead)
PLUS_MONTHS = [0, 6, 12, 24]

# =============================================================================
# FEATURE COLUMNS
# =============================================================================

# Base features that are always included
FEATURES_MONTH = ["visit_month"]

# =============================================================================
# DEFAULT PREDICTION VALUES
# =============================================================================

# Default values for patients with missing data (bad patients)
DEFAULT_PREDICTIONS = {
    "updrs_1": 3,  # Default UPDRS part 1 score
    "updrs_2": 1,  # Default UPDRS part 2 score
    "updrs_3": 1,  # Default UPDRS part 3 score
    "updrs_4": 0,  # UPDRS part 4 is always 0
}

# =============================================================================
# DATA PROCESSING PARAMETERS
# =============================================================================

# Minimum visit month threshold for categorizing patients
MIN_VISIT_MONTH_THRESHOLD = 6

# Column names for clinical data
CLINICAL_COLUMNS = ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]

# =============================================================================
# FILE NAMES
# =============================================================================

# Input CSV file names
TRAIN_CLINICAL_FILE = "train_clinical_data.csv"
TRAIN_PEPTIDES_FILE = "train_peptides.csv"
TRAIN_PROTEINS_FILE = "train_proteins.csv"
