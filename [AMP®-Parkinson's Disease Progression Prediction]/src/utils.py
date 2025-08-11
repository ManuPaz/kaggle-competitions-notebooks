"""
Utility functions for Parkinson's Disease Progression Prediction.

This module contains utility functions for data processing, feature engineering,
data preparation, and patient analysis.

Author: [Your Name]
Date: [Current Date]
"""

import types
from functools import reduce

import numpy as np
import pandas as pd
import pkg_resources
from sklearn.model_selection import train_test_split
from src.params import CLINICAL_COLUMNS, NUM_CANDIDATES, PLUS_MONTHS, RANDOM_STATE, TEST_SIZE, UPDRS_PARTS
from tqdm import tqdm


def get_imports():
    """
    Get a list of all imported packages and their versions.

    This function analyzes the global namespace to identify imported modules
    and retrieves their version information from pkg_resources.

    Returns:
        list: List of tuples containing (package_name, version)

    Example:
        >>> imports = get_imports()
        >>> for pkg, ver in imports:
        ...     print(f"{pkg}=={ver}")
    """
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            # Split ensures you get root package, not just imported function
            name = val.__name__.split(".")[0]
        elif isinstance(val, type):
            name = val.__module__.split(".")[0]

        # Some packages are weird and have different imported names vs. system names
        if name == "PIL":
            name = "Pillow"
        elif name == "sklearn":
            name = "scikit-learn"

        yield name


def get_requirements():
    """
    Get requirements list with package names and versions.

    Returns:
        list: List of tuples containing (package_name, version)
    """
    imports = list(set(get_imports()))

    requirements = []
    for m in pkg_resources.working_set:
        if m.project_name in imports and m.project_name != "pip":
            requirements.append((m.project_name, m.version))

    return requirements


def print_requirements():
    """
    Print all package requirements in pip format.

    Example output:
        pandas==1.5.3
        numpy==1.24.3
        xgboost==1.7.6
    """
    requirements = get_requirements()
    for r in requirements:
        print("{}=={}".format(*r))


def get_peptide_candidates(peptides_df, num_candidates=None):
    """
    Get peptide candidates based on coefficient of variation (CV) of abundance.

    This function calculates the coefficient of variation for peptide abundance
    across patients and time points, then selects the peptides with the highest
    variability as candidates for feature engineering.

    Args:
        peptides_df (pd.DataFrame): DataFrame containing peptide data with columns:
            - patient_id: Patient identifier
            - Peptide: Peptide sequence/identifier
            - PeptideAbundance: Abundance values
        num_candidates (int, optional): Number of peptide candidates to return.
            If None, uses the value from params.NUM_CANDIDATES.

    Returns:
        pd.Series: Series of peptide identifiers with highest CV values

    Example:
        >>> peptide_candidates = get_peptide_candidates(train_peptides, num_candidates=10)
        >>> print(f"Selected {len(peptide_candidates)} peptide candidates")
    """
    if num_candidates is None:
        num_candidates = NUM_CANDIDATES

    # Calculate the coefficient of variation (CV) for PeptideAbundance per patient_ids and Peptides
    train_peptides_df_agg = peptides_df[["patient_id", "Peptide", "PeptideAbundance"]]
    train_peptides_df_agg = train_peptides_df_agg.groupby(["patient_id", "Peptide"])["PeptideAbundance"].aggregate(
        ["mean", "std"]
    )

    # Calculate CV as percentage: (std/mean) * 100
    train_peptides_df_agg["CV_PeptideAbundance[%]"] = train_peptides_df_agg["std"] / train_peptides_df_agg["mean"] * 100

    # Calculate mean CV value across all patients for each peptide
    abundance_cv_mean = train_peptides_df_agg.groupby("Peptide")["CV_PeptideAbundance[%]"].mean().reset_index()

    # Sort by CV in descending order and select top candidates
    abundance_cv_mean = abundance_cv_mean.sort_values(by="CV_PeptideAbundance[%]", ascending=False).reset_index(
        drop=True
    )

    # Get peptide candidates
    peptide_candidates = abundance_cv_mean.loc[: num_candidates - 1, "Peptide"]
    return peptide_candidates


def get_protein_candidates(proteins_df, num_candidates=None):
    """
    Get protein candidates based on coefficient of variation (CV) of NPX values.

    This function calculates the coefficient of variation for protein NPX values
    across patients and time points, then selects the proteins with the highest
    variability as candidates for feature engineering.

    Args:
        proteins_df (pd.DataFrame): DataFrame containing protein data with columns:
            - patient_id: Patient identifier
            - UniProt: Protein identifier
            - NPX: Normalized protein expression values
        num_candidates (int, optional): Number of protein candidates to return.
            If None, uses the value from params.NUM_CANDIDATES.

    Returns:
        pd.Series: Series of protein identifiers (UniProt) with highest CV values

    Example:
        >>> protein_candidates = get_protein_candidates(train_proteins, num_candidates=10)
        >>> print(f"Selected {len(protein_candidates)} protein candidates")
    """
    if num_candidates is None:
        num_candidates = NUM_CANDIDATES

    # Calculate the coefficient of variation (CV) for NPX per patient_ids and UniProt
    train_proteins_df_agg = proteins_df[["patient_id", "UniProt", "NPX"]]
    train_proteins_df_agg = train_proteins_df_agg.groupby(["patient_id", "UniProt"])["NPX"].aggregate(["mean", "std"])

    # Calculate CV as percentage: (std/mean) * 100
    train_proteins_df_agg["CV_NPX[%]"] = train_proteins_df_agg["std"] / train_proteins_df_agg["mean"] * 100

    # Calculate mean CV value across all patients for each protein
    NPX_cv_mean = train_proteins_df_agg.groupby("UniProt")["CV_NPX[%]"].mean().reset_index()

    # Sort by CV in descending order and select top candidates
    NPX_cv_mean = NPX_cv_mean.sort_values(by="CV_NPX[%]", ascending=False).reset_index(drop=True)

    # Get protein candidates
    protein_candidates = NPX_cv_mean.loc[: num_candidates - 1, "UniProt"]
    return protein_candidates


def preprocessing_data(
    clinical_df,
    peptides_df,
    proteins_df,
    peptide_candidates,
    protein_candidates,
    train=True,
):
    """
    Preprocess and merge clinical, peptide, and protein data.

    This function performs the following steps:
    1. Filters peptides and proteins to selected candidates
    2. Pivots peptide and protein data to wide format
    3. Merges all data sources on visit_id
    4. Handles missing features by adding NaN columns
    5. Sorts data by patient_id and visit_month

    Args:
        clinical_df (pd.DataFrame): Clinical data with visit information
        peptides_df (pd.DataFrame): Peptide abundance data
        proteins_df (pd.DataFrame): Protein NPX data
        peptide_candidates (pd.Series): Selected peptide identifiers
        protein_candidates (pd.Series): Selected protein identifiers
        train (bool): If True, sorts by patient_id and visit_month.
                     If False, sorts by patient_id, updrs_test, and visit_month.

    Returns:
        tuple: (processed_dataframe, feature_columns, all_features)
            - processed_dataframe: Merged and processed DataFrame
            - feature_columns: List of peptide and protein feature columns
            - all_features: List of all features including visit_month

    Example:
        >>> df, features_ewn, features = preprocessing_data(
        ...     clinical_df, peptides_df, proteins_df,
        ...     peptide_candidates, protein_candidates, train=True
        ... )
    """
    # Filter peptides to selected candidates and pivot to wide format
    peptides_df_use = peptides_df.loc[peptides_df.Peptide.isin(peptide_candidates)].reset_index(drop=True)
    peptides_df_use = peptides_df_use.pivot_table(
        index=["visit_id"],
        columns=["Peptide"],
        values="PeptideAbundance",
        fill_value=np.nan,
    )

    # Filter proteins to selected candidates and pivot to wide format
    proteins_df_use = proteins_df.loc[proteins_df.UniProt.isin(protein_candidates)].reset_index(drop=True)
    proteins_df_use = proteins_df_use.pivot_table(
        index=["visit_id"], columns=["UniProt"], values="NPX", fill_value=np.nan
    )

    # Merge all data sources
    df = pd.merge(clinical_df, proteins_df_use, on="visit_id", how="left")
    df = pd.merge(df, peptides_df_use, on="visit_id", how="left")

    # Define feature columns
    FEATURES_ewn = list(peptide_candidates) + list(protein_candidates)
    FEATURES = FEATURES_ewn + ["visit_month"]

    # Add missing features as NaN columns
    for feature in FEATURES_ewn:
        if feature not in df.columns:
            print(f"Feature {feature} not in columns, adding it as NA")
            df[feature] = np.nan

    # Sort data based on training or inference mode
    if train:
        df = df.sort_values(by=["patient_id", "visit_month"])
        # Note: Forward fill commented out in original code
        # df[FEATURES_ewn] = df.groupby("patient_id", sort=False).fillna(method="ffill")[FEATURES_ewn]
    else:
        df = df.sort_values(by=["patient_id", "updrs_test", "visit_month"])
        # Note: Forward fill commented out in original code
        # df[FEATURES_ewn] = df.groupby(["patient_id", "updrs_test"], sort=False).fillna(method="ffill")[FEATURES_ewn]

    return df, FEATURES_ewn, FEATURES


def create_X_y_train_dataset(df, updrs_part, plus_month):
    """
    Create training dataset for a specific UPDRS part and time horizon.

    This function creates feature matrix X and target vector y for training
    models to predict UPDRS scores at a specific time horizon (plus_month)
    from baseline measurements.

    Args:
        df (pd.DataFrame): Preprocessed training data
        updrs_part (int): UPDRS part to predict (1, 2, 3, or 4)
        plus_month (int): Time horizon in months (0, 6, 12, or 24)

    Returns:
        tuple: (X, y)
            - X: Feature matrix with visit_id as index
            - y: Target vector with patient_id, visit_id, and target UPDRS score

    Example:
        >>> X, y = create_X_y_train_dataset(train_df, updrs_part=1, plus_month=6)
        >>> print(f"X shape: {X.shape}, y shape: {y.shape}")
    """
    # Remove rows with missing target values
    df_ = df.dropna(subset=[f"updrs_{updrs_part}"])

    X_visit_ids = []
    y_visit_ids = []
    patient_ids = df["patient_id"].unique()

    # For each patient, create visit_id pairs for baseline and target time
    for i, patient_id in enumerate(patient_ids):
        patient_df = df_[df_["patient_id"] == patient_id]

        # Calculate target months (baseline + plus_month)
        plus_months = patient_df["visit_month"] + plus_month
        plus_months = patient_df.query("visit_month in @plus_months")["visit_month"]

        # Calculate baseline months
        original_months = plus_months - plus_month

        patient_id = str(patient_id)

        # Create visit_id strings for baseline (X) and target (y)
        X_visit_id = [patient_id + "_" + str(original_month) for original_month in original_months]
        y_visit_id = [patient_id + "_" + str(plus_month) for plus_month in plus_months]

        X_visit_ids.extend(X_visit_id)
        y_visit_ids.extend(y_visit_id)

    # Create feature matrix X (baseline data)
    X = df_.query("visit_id in @X_visit_ids")
    X = X.drop(CLINICAL_COLUMNS, axis=1)  # Remove UPDRS target columns
    X.reset_index(drop=True, inplace=True)

    # Create target vector y
    y = df_.query("visit_id in @y_visit_ids")
    y = y[["patient_id", "visit_id", f"updrs_{updrs_part}"]]
    y.reset_index(drop=True, inplace=True)

    return X, y


def create_X_y_dict(df):
    """
    Create dictionaries of feature matrices and target vectors for all UPDRS parts and time horizons.

    This function creates training datasets for all combinations of:
    - UPDRS parts: 1, 2, 3, 4
    - Time horizons: 0, 6, 12, 24 months

    Args:
        df (pd.DataFrame): Preprocessed training data

    Returns:
        tuple: (X_dict, y_dict)
            - X_dict: Dictionary with keys like "updrs_1_plus_6_months" and values as feature matrices
            - y_dict: Dictionary with same keys and values as target vectors

    Example:
        >>> X_dict, y_dict = create_X_y_dict(train_df)
        >>> print(f"Created {len(X_dict)} training datasets")
        >>> for key in X_dict.keys():
        ...     print(f"{key}: X={X_dict[key].shape}, y={y_dict[key].shape}")
    """
    X_dict = {}
    y_dict = {}

    # Create datasets for all UPDRS parts and time horizons
    for updrs_part in tqdm(UPDRS_PARTS, desc="Creating datasets"):
        for plus_month in PLUS_MONTHS:
            X, y = create_X_y_train_dataset(df, updrs_part, plus_month)
            key = f"updrs_{updrs_part}_plus_{plus_month}_months"
            X_dict[key] = X
            y_dict[key] = y

    return X_dict, y_dict


def analyze_patients(X_dict):
    """
    Analyze patient data availability across all training datasets.

    This function analyzes which patients have complete data across all
    UPDRS parts and time horizons, identifying patients with good vs. bad data quality.

    Args:
        X_dict (dict): Dictionary of feature matrices from create_X_y_dict

    Returns:
        tuple: (intersection, union, patients_array)
            - intersection: List of patients with complete data across all datasets
            - union: List of all unique patients
            - patients_array: List of patient arrays for each dataset

    Example:
        >>> intersection, union, patients_array = analyze_patients(X_dict)
        >>> print(f"Patients with complete data: {len(intersection)}")
        >>> print(f"Total unique patients: {len(union)}")
    """
    patients_array = []

    # Collect patient lists from each dataset
    for key, X in X_dict.items():
        print(
            key,
            X_dict[key].shape,
            len(X_dict[key].patient_id.unique()),
            X_dict[key].visit_month.unique(),
        )
        patients = X.patient_id.unique()
        patients_array.append(patients)

    # Find intersection (patients with complete data) and union (all patients)
    intersection = reduce(lambda a, b: set(a).intersection(set(b)), patients_array)
    union = reduce(lambda a, b: set(a).union(set(b)), patients_array)

    intersection = list(intersection)
    intersection.sort()

    return intersection, union, patients_array


def create_train_test_split(X_dict):
    """
    Create train-test split for patient IDs.

    This function splits patients into training and testing sets,
    ensuring that the same patients are used across all datasets.

    Args:
        X_dict (dict): Dictionary of feature matrices

    Returns:
        tuple: (patients_train, patients_test)
            - patients_train: List of patient IDs for training
            - patients_test: List of patient IDs for testing

    Example:
        >>> patients_train, patients_test = create_train_test_split(X_dict)
        >>> print(f"Training patients: {len(patients_train)}")
        >>> print(f"Testing patients: {len(patients_test)}")
    """
    # Use the first dataset for patient splitting (all should have same patients)
    patients_train, patients_test = train_test_split(
        X_dict["updrs_1_plus_0_months"].patient_id.unique(), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    return patients_train, patients_test


def categorize_patients(patients_train, intersection, X_dict, train_df):
    """
    Categorize patients into 'good' and 'bad' based on data completeness.

    This function categorizes patients into two groups:
    - 'good': Patients with complete data across all time points
    - 'bad': Patients with missing data at some time points

    Args:
        patients_train (list): List of training patient IDs
        intersection (list): List of patients with complete data
        X_dict (dict): Dictionary of feature matrices
        train_df (pd.DataFrame): Original training dataframe

    Returns:
        tuple: (patients_train_good, patients_train_bad, patients_good)
            - patients_train_good: Training patients with complete data
            - patients_train_bad: Training patients with missing data
            - patients_good: All patients with complete data

    Example:
        >>> good_train, bad_train, all_good = categorize_patients(
        ...     patients_train, intersection, X_dict, train_df
        ... )
    """
    # Categorize training patients
    patients_train_good = list(set(patients_train).intersection(set(intersection)))
    patients_train_bad = list(set(patients_train) - (set(intersection)))

    # All patients with complete data
    patients_good = list(set(X_dict["updrs_1_plus_0_months"].patient_id.unique()).intersection(intersection))

    return patients_train_good, patients_train_bad, patients_good


def categorize_test_patients(patients_test, train_df, test_df):
    """
    Categorize test patients into 'good' and 'bad' based on data availability.

    This function analyzes test patients to determine which ones have
    complete data at month 6, categorizing them for different prediction strategies.

    Args:
        patients_test (list): List of test patient IDs
        train_df (pd.DataFrame): Training dataframe for reference
        test_df (pd.DataFrame): Test dataframe to analyze

    Returns:
        tuple: (patients_test_good, patients_test_bad)
            - patients_test_good: Test patients with complete data
            - patients_test_bad: Test patients with missing data

    Example:
        >>> test_good, test_bad = categorize_test_patients(
        ...     patients_test, train_df, test_df
        ... )
    """
    # Analyze test patient data availability
    groups_test = train_df.loc[train_df.patient_id.isin(patients_test)].groupby(["patient_id", "visit_month"]).size()

    # Create complete index for all patient-month combinations
    index = pd.MultiIndex.from_product(
        [groups_test.index.get_level_values(0), groups_test.index.get_level_values(1)],
        names=groups_test.index.names,
    )

    # Reindex and fill missing combinations with 0
    groups_test = groups_test.reindex(index, fill_value=0).reset_index().rename(columns={0: "count"})

    # Identify patients missing data at month 6
    patients_test_bad = groups_test.query("visit_month==6 and count==0").patient_id.unique()
    patients_test_good = list(set(patients_test) - set(patients_test_bad))

    return patients_test_good, patients_test_bad
