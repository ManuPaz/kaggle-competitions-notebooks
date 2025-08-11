"""
Data Processing and Loading Functions

This module contains functions for loading, preprocessing, and preparing the training
data for the LLM Science Exam model, including data augmentation and cross-validation.
These functions handle the complete data pipeline from raw CSV files to
Hugging Face datasets ready for training.

Functions:
    load_training_data: Load and prepare training data from CSV files
    prepare_cross_validation_data: Prepare cross-validation splits of the data
    prepare_final_training_data: Prepare final training and validation datasets
    create_datasets: Convert pandas dataframes to Hugging Face datasets
    add_combined_text_column: Add a combined text column for analysis
    get_removable_columns: Get list of columns that should be removed during tokenization
    tokenize_dataset: Tokenize a dataset using the provided preprocessing function
"""

import os
from typing import List, Tuple

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold


def load_training_data(data_path: str, use_sample: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare training data from CSV files.
    
    This function loads both the original training data and augmented data,
    performs initial cleaning, and prepares them for further processing.
    The augmented data is filtered to avoid duplicates with the original data.

    Args:
        data_path: Path to the data directory containing CSV files
        use_sample: Whether to use a sample of the data for testing (currently unused)

    Returns:
        tuple: A tuple containing:
            - augmented_df: Augmented training dataframe with context
            - df: Original training dataframe
            
    Note:
        The function expects two CSV files in the data_path:
        - train_with_context2.csv: Original training data
        - all_12_with_context2.csv: Augmented training data
        
    Example:
        >>> data_path = "data/llm_science_exam"
        >>> augmented_df, original_df = load_training_data(data_path)
        >>> print(f"Augmented data shape: {augmented_df.shape}")
        >>> print(f"Original data shape: {original_df.shape}")
    """
    # Load original training data
    df = pd.read_csv(os.path.join(data_path, "train_with_context2.csv"))

    # Load augmented data
    augmented_df = pd.read_csv(os.path.join(data_path, "all_12_with_context2.csv"))

    # Drop 'id' column if it exists
    if "id" in df.columns:
        df = df.drop("id", axis=1)

    # Remove duplicates and prepare augmented data
    augmented_df = augmented_df.loc[~augmented_df.prompt.isin(df.prompt)].dropna()
    augmented_df = augmented_df.drop_duplicates(subset="prompt")

    # Select required columns
    required_columns = ["prompt", "context", "A", "B", "C", "D", "E", "answer"]
    augmented_df = augmented_df[required_columns]

    return augmented_df, df


def prepare_cross_validation_data(
    augmented_df: pd.DataFrame, n_splits: int = 5, random_state: int = 21
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Prepare cross-validation splits of the data.
    
    This function creates stratified cross-validation splits based on the
    answer distribution, ensuring that each fold has a representative
    distribution of answer choices.

    Args:
        augmented_df: Augmented training dataframe
        n_splits: Number of CV splits (default: 5)
        random_state: Random seed for reproducibility (default: 21)

    Returns:
        list: List of (train, val) dataframe tuples, where each tuple contains:
            - train_df: Training data for that fold
            - val_df: Validation data for that fold
            
    Example:
        >>> cv_splits = prepare_cross_validation_data(augmented_df, n_splits=3)
        >>> for i, (train_df, val_df) in enumerate(cv_splits):
        ...     print(f"Fold {i}: Train={train_df.shape}, Val={val_df.shape}")
    """
    kfold = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    cv_splits = []
    for train_indices, val_indices in kfold.split(augmented_df, augmented_df.answer):
        train_df = augmented_df.iloc[train_indices]
        val_df = augmented_df.iloc[val_indices]
        cv_splits.append((train_df, val_df))

    return cv_splits


def prepare_final_training_data(
    augmented_df: pd.DataFrame, original_df: pd.DataFrame, fold: int = None, n_splits: int = 5, random_state: int = 21
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare final training and validation datasets.
    
    This function prepares the final training and validation datasets based on
    the specified fold or uses all augmented data. It performs cleaning operations
    such as handling missing values and removing duplicates.

    Args:
        augmented_df: Augmented training dataframe
        original_df: Original training dataframe (used as validation)
        fold: Specific fold to use (None for all data, default: None)
        n_splits: Number of CV splits (default: 5)
        random_state: Random seed for reproducibility (default: 21)

    Returns:
        tuple: A tuple containing:
            - train_df: Final training dataframe (cleaned and prepared)
            - val_df: Final validation dataframe (original data)
            
    Note:
        If fold is specified, only that specific fold is used for training.
        If fold is None, all augmented data is used for training.
        The original data is always used as validation.
        
    Example:
        >>> # Use specific fold
        >>> train_df, val_df = prepare_final_training_data(
        ...     augmented_df, original_df, fold=0, n_splits=5
        ... )
        >>> 
        >>> # Use all augmented data
        >>> train_df, val_df = prepare_final_training_data(
        ...     augmented_df, original_df, fold=None
        ... )
    """
    if fold is not None:
        # Use specific fold
        kfold = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        for f, (train_indices, val_indices) in enumerate(kfold.split(augmented_df, augmented_df.answer)):
            if f == fold:
                train_df = augmented_df.iloc[train_indices]
                break
    else:
        # Use all augmented data
        train_df = augmented_df.copy()

    # Clean and prepare training data
    train_df = train_df.fillna("MASK_NAS")
    train_df = train_df.loc[~train_df.prompt.isin(original_df.prompt)]
    train_df = train_df.drop_duplicates(subset="prompt")

    # Use original data as validation
    val_df = original_df.copy()

    return train_df, val_df


def create_datasets(
    train_df: pd.DataFrame, val_df: pd.DataFrame, use_sample: bool = True, sample_size: int = 100
) -> Tuple[Dataset, Dataset]:
    """
    Convert pandas dataframes to Hugging Face datasets.
    
    This function converts pandas dataframes to Hugging Face Dataset objects,
    which are required for training with the Transformers library. It also
    supports sampling for development and testing purposes.

    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        use_sample: Whether to use a sample of training data (default: True)
        sample_size: Size of the sample if use_sample is True (default: 100)

    Returns:
        tuple: A tuple containing:
            - train_ds: Training dataset as Hugging Face Dataset
            - val_ds: Validation dataset as Hugging Face Dataset
            
    Example:
        >>> train_ds, val_ds = create_datasets(
        ...     train_df, val_df, use_sample=True, sample_size=500
        ... )
        >>> print(f"Training dataset: {train_ds}")
        >>> print(f"Validation dataset: {val_ds}")
    """
    if use_sample:
        train_ds = Dataset.from_pandas(train_df.iloc[:sample_size])
    else:
        train_ds = Dataset.from_pandas(train_df)

    val_ds = Dataset.from_pandas(val_df)

    return train_ds, val_ds


def add_combined_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a combined text column for analysis.
    
    This function concatenates all text fields (prompt and answer options)
    into a single column, which can be useful for text analysis, feature
    engineering, or debugging purposes.

    Args:
        df: Input dataframe with columns: prompt, A, B, C, D, E

    Returns:
        pd.DataFrame: Dataframe with an additional 'all' column containing
                     concatenated text from all fields
                     
    Example:
        >>> df_with_combined = add_combined_text_column(df)
        >>> print(df_with_combined['all'].iloc[0][:100])  # First 100 chars
    """
    df["all"] = df["prompt"] + df["A"] + df["B"] + df["C"] + df["D"] + df["E"]

    return df


def get_removable_columns(dataset: Dataset) -> List[str]:
    """
    Get list of columns that should be removed during tokenization.
    
    This function identifies which columns from the original dataset should
    be removed after tokenization, as they are no longer needed for
    model training.

    Args:
        dataset: Input dataset (Hugging Face Dataset object)

    Returns:
        list: List of column names that should be removed during tokenization
        
    Note:
        The function checks for the presence of each column in the dataset
        before including it in the removable list, ensuring compatibility
        with different dataset structures.
        
    Example:
        >>> removable_cols = get_removable_columns(dataset)
        >>> print(f"Columns to remove: {removable_cols}")
        >>> tokenized_ds = dataset.map(preprocess_func, remove_columns=removable_cols)
    """
    required_columns = ["prompt", "A", "B", "C", "D", "E", "answer", "__index_level_0__", "context", "source"]

    removable_cols = [col for col in required_columns if col in dataset.features]
    return removable_cols


def tokenize_dataset(
    dataset: Dataset, tokenizer, preprocess_func, max_length: int, remove_columns: List[str]
) -> Dataset:
    """
    Tokenize a dataset using the provided preprocessing function.
    
    This function applies tokenization to a dataset using the specified
    preprocessing function and removes unnecessary columns. It's a wrapper
    around the dataset.map() method for consistency and clarity.

    Args:
        dataset: Input dataset (Hugging Face Dataset object)
        tokenizer: Tokenizer to use (passed to preprocess_func)
        preprocess_func: Preprocessing function that handles tokenization
        max_length: Maximum input length (passed to preprocess_func)
        remove_columns: List of column names to remove after tokenization

    Returns:
        Dataset: Tokenized dataset ready for model training
        
    Example:
        >>> preprocess_func = preprocess_wrapper(tokenizer, max_input=450)
        >>> remove_cols = get_removable_columns(dataset)
        >>> tokenized_ds = tokenize_dataset(
        ...     dataset, tokenizer, preprocess_func, 450, remove_cols
        ... )
    """
    tokenized_ds = dataset.map(preprocess_func, batched=False, remove_columns=remove_columns)

    return tokenized_ds

