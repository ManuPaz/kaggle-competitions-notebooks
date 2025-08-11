"""
Evaluation metrics and model training functions for Parkinson's Disease Progression Prediction.

This module contains functions for evaluating model performance using SMAPE metric
and training models for different patient categories.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import pandas as pd
import xgboost
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from src.params import DEFAULT_PREDICTIONS, FEATURES_MONTH, MIN_VISIT_MONTH_THRESHOLD, MODELS_DICT


def smape1(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).

    SMAPE is a metric that measures the accuracy of predictions as a percentage.
    It's symmetric and handles cases where actual values can be zero.
    The formula used is: 100 * mean(|y_true - y_pred| / ((|y_true| + |y_pred|) / 2))

    Args:
        y_true (np.array): Ground truth values
        y_pred (np.array): Predicted values

    Returns:
        float: SMAPE score (lower is better, 0 is perfect)

    Example:
        >>> y_true = np.array([1, 2, 3, 4, 5])
        >>> y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        >>> score = smape1(y_true, y_pred)
        >>> print(f"SMAPE score: {score:.2f}")

    Note:
        This implementation adds 1 to both y_true and y_pred before calculation
        to handle cases where values might be 0, which would cause division by zero.
    """
    # Add 1 to avoid division by zero issues
    y_true = y_true + 1
    y_pred = y_pred + 1

    # Calculate numerator and denominator
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    # Handle cases where both y_true and y_pred are 0
    positive_index = (y_true != 0) | (y_pred != 0)

    # Initialize SMAPE array
    smape = np.zeros(len(y_true))

    # Calculate SMAPE only for non-zero cases
    smape[positive_index] = numerator[positive_index] / denominator[positive_index]

    # Return mean SMAPE as percentage
    smape = 100 * np.mean(smape)
    return smape


def get_model_instance(model_name, params=None):
    """
    Get a model instance with specified parameters.

    This function creates and returns a model instance based on the model name
    and parameters. It supports XGBoost, Linear Regression, and SVR models.

    Args:
        model_name (str): Name of the model to create. Options: "xgboost", "linear", "svr"
        params (dict, optional): Model parameters. If None, uses default parameters from MODELS_DICT

    Returns:
        model: Trained model instance

    Example:
        >>> model = get_model_instance("xgboost", {"n_estimators": 100})
        >>> print(f"Model type: {type(model)}")

    Raises:
        ValueError: If model_name is not supported
    """
    if model_name not in MODELS_DICT:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(MODELS_DICT.keys())}")

    if params is None:
        params = MODELS_DICT[model_name]["params"]

    model_class = MODELS_DICT[model_name]["model"]

    if model_name == "xgboost":
        return xgboost.XGBRegressor(**params)
    elif model_name == "linear":
        return LinearRegression(**params)
    elif model_name == "svr":
        return SVR(**params)
    else:
        raise ValueError(f"Model instantiation not implemented for: {model_name}")


def train_models(X_dict, y_dict, patients_train_good, patients_train_bad, patients_good):
    """
    Train models for both 'good' and 'bad' patient categories.

    This function trains separate models for patients with complete data (good)
    and patients with missing data (bad). It uses different features and
    training strategies for each category.

    Args:
        X_dict (dict): Dictionary of feature matrices for all UPDRS parts and time horizons
        y_dict (dict): Dictionary of target vectors for all UPDRS parts and time horizons
        patients_train_good (list): List of training patient IDs with complete data
        patients_train_bad (list): List of training patient IDs with missing data
        patients_good (list): List of all patient IDs with complete data

    Returns:
        dict: Dictionary containing trained models for both categories

    Example:
        >>> models = train_models(X_dict, y_dict, patients_train_good, patients_train_bad, patients_good)
        >>> print(f"Trained {len(models['good'])} good models and {len(models['bad'])} bad models")
    """
    models = {"bad": {}, "good": {}}

    # Use SVR model for this training session
    MODEL_USE = "svr"
    params = {"kernel": "rbf", "degree": 2, "C": 2, "epsilon": 2}
    FEATURES = FEATURES_MONTH

    print("Training models for all UPDRS parts and time horizons...")

    for key, X in X_dict.items():
        target = key.rsplit("_", 3)[0]  # Extract UPDRS part (e.g., "updrs_1")
        y = y_dict[key]

        # Add visit_month to target dataframe
        y["visit_month"] = y["visit_id"].transform(lambda x: x.split("_")[1]).astype(int)

        print(f"\nTraining model for {key}...")

        # Train model for good patients (complete data)
        model = get_model_instance(MODEL_USE, params)
        X_train_good = X.loc[X.patient_id.isin(patients_train_good)]
        y_train_good = y.loc[y.patient_id.isin(patients_train_good)]

        if len(X_train_good) > 0:
            model.fit(X_train_good[FEATURES], y_train_good[target])
            print(f"  Good patients: {len(X_train_good)} samples")
        else:
            print(f"  Warning: No good patients for training {key}")

        # Train model for bad patients (missing data)
        model_bad = get_model_instance(MODEL_USE, params)
        X_train_bad = X.loc[X.patient_id.isin(patients_train_bad)]
        y_train_bad = y.loc[y.patient_id.isin(patients_train_bad)]

        if len(X_train_bad) > 0:
            model_bad.fit(X_train_bad[FEATURES], y_train_bad[target])
            print(f"  Bad patients: {len(X_train_bad)} samples")
        else:
            print(f"  Warning: No bad patients for training {key}")

        # Store models
        models["good"][key] = model
        models["bad"][key] = model_bad

    return models


def evaluate_models(X_dict, y_dict, models, patients_test_good, patients_test_bad):
    """
    Evaluate trained models on test data.

    This function evaluates the performance of trained models on test data,
    calculating SMAPE scores for both good and bad patient categories.

    Args:
        X_dict (dict): Dictionary of feature matrices for all UPDRS parts and time horizons
        y_dict (dict): Dictionary of target vectors for all UPDRS parts and time horizons
        models (dict): Dictionary containing trained models for both categories
        patients_test_good (list): List of test patient IDs with complete data
        patients_test_bad (list): List of test patient IDs with missing data

    Returns:
        pd.DataFrame: DataFrame containing predictions and evaluation results

    Example:
        >>> results_df = evaluate_models(X_dict, y_dict, models, patients_test_good, patients_test_bad)
        >>> print(f"Evaluation results shape: {results_df.shape}")
    """
    df_pred = None
    FEATURES = FEATURES_MONTH

    print("\nEvaluating models on test data...")

    for key, X in X_dict.items():
        target = key.rsplit("_", 3)[0]  # Extract UPDRS part
        y = y_dict[key]
        y["visit_month"] = y["visit_id"].transform(lambda x: x.split("_")[1]).astype(int)

        print(f"\nEvaluating {key}...")

        # Evaluate on good test patients
        X_test_good = X.loc[(X.patient_id.isin(patients_test_good)) | (X.visit_month <= MIN_VISIT_MONTH_THRESHOLD)]
        y_test_good = y.loc[(y.patient_id.isin(patients_test_good)) | (X.visit_month <= MIN_VISIT_MONTH_THRESHOLD)]

        if len(X_test_good) > 0:
            model_good = models["good"][key]
            y_pred_good = model_good.predict(X_test_good[FEATURES])

            # Ensure predictions are non-negative and rounded
            y_pred_good[y_pred_good < 0] = 0
            y_pred_good = np.round(y_pred_good)

            # Calculate SMAPE score
            score = smape1(y_test_good[target], y_pred_good)
            print(f"  Good patients SMAPE: {score:.2f} (samples: {len(X_test_good)})")

            # Store predictions
            df_pred_good = pd.DataFrame(
                {"real": y_test_good[target], "pred": y_pred_good, "key": target, "category": "good"}
            )
            df_pred = pd.concat([df_pred, df_pred_good]) if df_pred is not None else df_pred_good

        # Evaluate on bad test patients (using default values)
        X_test_bad = X.loc[(X.patient_id.isin(patients_test_bad)) & (X.visit_month > MIN_VISIT_MONTH_THRESHOLD)]
        y_test_bad = y.loc[(y.patient_id.isin(patients_test_bad)) & (X.visit_month > MIN_VISIT_MONTH_THRESHOLD)]

        if len(X_test_bad) > 0:
            # Use default predictions for bad patients
            if target == "updrs_1":
                y_pred_bad = DEFAULT_PREDICTIONS["updrs_1"]
            elif target == "updrs_2":
                y_pred_bad = DEFAULT_PREDICTIONS["updrs_2"]
            elif target == "updrs_3":
                y_pred_bad = DEFAULT_PREDICTIONS["updrs_3"]
            else:
                y_pred_bad = DEFAULT_PREDICTIONS.get(target, 0)

            # Calculate SMAPE score
            score = smape1(y_test_bad[target], y_pred_bad)
            print(f"  Bad patients SMAPE: {score:.2f} (samples: {len(X_test_bad)})")

            # Store predictions
            df_pred_bad = pd.DataFrame(
                {"real": y_test_bad[target], "pred": y_pred_bad, "key": target, "category": "bad"}
            )
            df_pred = pd.concat([df_pred, df_pred_bad])

    return df_pred


def get_final_predictions(df_pred):
    """
    Generate final predictions and calculate overall SMAPE score.

    This function processes the evaluation results to generate final predictions
    and calculate the overall model performance.

    Args:
        df_pred (pd.DataFrame): DataFrame containing predictions and evaluation results

    Returns:
        tuple: (final_predictions, overall_smape)
            - final_predictions: Processed predictions DataFrame
            - overall_smape: Overall SMAPE score across all predictions

    Example:
        >>> final_pred, smape = get_final_predictions(df_pred)
        >>> print(f"Overall SMAPE: {smape:.2f}")
    """
    if df_pred is None:
        print("No predictions to process")
        return None, None

    # Set UPDRS part 4 predictions to 0 (as per competition rules)
    df_pred.loc[df_pred.key == "updrs_4", "pred"] = 0

    # Calculate overall SMAPE
    overall_smape = smape1(df_pred["real"], df_pred["pred"])

    print("\nFinal Results:")
    print(f"Overall SMAPE: {overall_smape:.2f}")
    print(f"Total predictions: {len(df_pred)}")

    # Print breakdown by UPDRS part
    for key in df_pred["key"].unique():
        key_data = df_pred[df_pred["key"] == key]
        key_smape = smape1(key_data["real"], key_data["pred"])
        print(f"  {key}: SMAPE = {key_smape:.2f} ({len(key_data)} samples)")

    return df_pred, overall_smape


def predict_test_data(test_df, models, peptide_candidates, protein_candidates):
    """
    Generate predictions for test data using trained models.

    This function applies trained models to new test data, categorizing patients
    and generating predictions using appropriate strategies for each category.

    Args:
        test_df (pd.DataFrame): Preprocessed test data
        models (dict): Dictionary containing trained models for both categories
        peptide_candidates (pd.Series): Selected peptide identifiers
        protein_candidates (pd.Series): Selected protein identifiers

    Returns:
        pd.DataFrame: DataFrame containing predictions for test data

    Example:
        >>> predictions = predict_test_data(test_df, models, peptide_candidates, protein_candidates)
        >>> print(f"Generated {len(predictions)} predictions")
    """
    # Categorize test patients
    if test_df.visit_month.unique().min() > MIN_VISIT_MONTH_THRESHOLD:
        # Analyze test patient data availability
        groups_test = test_df.groupby(["patient_id", "visit_month"]).size()

        # Create complete index for all patient-month combinations
        index = pd.MultiIndex.from_product(
            [
                groups_test.index.get_level_values(0),
                groups_test.index.get_level_values(1),
            ],
            names=groups_test.index.names,
        )

        # Reindex and fill missing combinations with 0
        groups_test = groups_test.reindex(index, fill_value=0).reset_index().rename(columns={0: "count"})

        # Identify patients missing data at month 6
        patients_test_bad = groups_test.query("count==0 and visit_month==6").patient_id.unique()
        patients_test_good = list(set(test_df.patient_id.unique()) - set(patients_test_bad))
    else:
        patients_test_good = test_df.patient_id.unique()
        patients_test_bad = []

    print(f"Test patients - Good: {len(patients_test_good)}, Bad: {len(patients_test_bad)}")

    # Generate predictions for each UPDRS part
    predictions = []
    FEATURES = FEATURES_MONTH

    for updrs_part in [1, 2, 3, 4]:
        if updrs_part == 4:
            # UPDRS part 4 is always 0
            print("UPDRS part 4: Setting all predictions to 0")
            continue

        # Generate predictions for different time horizons
        for plus_month in [0, 6, 12, 24]:
            key = f"updrs_{updrs_part}_plus_{plus_month}_months"

            if key not in models["good"]:
                print(f"Warning: Model {key} not found in trained models")
                continue

            # Get test data for this combination
            test_data = test_df[test_df["updrs_test"] == f"updrs_{updrs_part}"]

            if len(test_data) == 0:
                continue

            # Predict for good patients
            test_good = test_data[test_data.patient_id.isin(patients_test_good)]
            if len(test_good) > 0:
                model_good = models["good"][key]
                pred_good = model_good.predict(test_good[FEATURES])

                # Ensure predictions are non-negative and rounded
                pred_good[pred_good < 0] = 0
                pred_good = np.round(pred_good)

                # Store predictions
                for i, (_, row) in enumerate(test_good.iterrows()):
                    predictions.append(
                        {
                            "patient_id": row["patient_id"],
                            "visit_month": row["visit_month"],
                            "updrs_test": f"updrs_{updrs_part}",
                            "plus_month": plus_month,
                            "prediction": pred_good[i],
                            "category": "good",
                        }
                    )

            # Predict for bad patients (use default values)
            test_bad = test_data[test_data.patient_id.isin(patients_test_bad)]
            if len(test_bad) > 0:
                default_pred = DEFAULT_PREDICTIONS[f"updrs_{updrs_part}"]

                for _, row in test_bad.iterrows():
                    predictions.append(
                        {
                            "patient_id": row["patient_id"],
                            "visit_month": row["visit_month"],
                            "updrs_test": f"updrs_{updrs_part}",
                            "plus_month": plus_month,
                            "prediction": default_pred,
                            "category": "bad",
                        }
                    )

    return pd.DataFrame(predictions)
