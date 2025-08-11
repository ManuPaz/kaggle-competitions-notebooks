"""
LLM Science Exam Training Pipeline

This package contains all the modules needed for training the LLM Science Exam model,
including data processing, model setup, training, and evaluation.

Modules:
    - constants: Configuration constants and paths
    - params: Training parameters and hyperparameters
    - utils: Utility functions and data collators
    - metrics: Evaluation metrics and scoring functions
    - data_processing: Data loading and preprocessing functions
    - model: Model setup and training configuration
"""

# Import main functions for easy access
from .constants import (
    CHECKPOINT_DIR,
    DATA_PATH,
    FREEZE_EMBEDDINGS,
    FREEZE_LAYERS,
    INDICES,
    MAX_INPUT_LENGTH,
    MAX_INPUT_LENGTH_VAL,
    MODEL_DIR,
    MODEL_NAME,
    N_SPLITS,
    OPTIONS,
    OUTPUT_DIR,
    SEED,
    STOP_WORDS,
    USE_SAMPLE,
)
from .data_processing import (
    add_combined_text_column,
    create_datasets,
    get_removable_columns,
    load_training_data,
    prepare_cross_validation_data,
    prepare_final_training_data,
    tokenize_dataset,
)
from .metrics import calculate_accuracy_at_k, competition_score, compute_metrics, map3_torch, predictions_to_map_output
from .model import (
    configure_model_freezing,
    get_model_info,
    load_trained_model,
    save_model_and_tokenizer,
    setup_model_and_tokenizer,
    setup_trainer,
    setup_training_arguments,
)
from .params import CV_CONFIG, DATA_CONFIG, EARLY_STOPPING, MODEL_FREEZING, TOKENIZATION_CONFIG
from .utils import (
    DataCollatorForMultipleChoice,
    create_option_mappings,
    preprocess_wrapper,
    set_random_seed,
    setup_device,
)

# Version information
__version__ = "1.0.0"
__author__ = "Competition Team"
__description__ = "LLM Science Exam Training Pipeline"


# Convenience function to get all imports
def get_all_imports():
    """
    Get a dictionary of all available imports for easy access.

    Returns:
        dict: Dictionary containing all available imports
    """
    return {
        "constants": [
            DATA_PATH,
            MODEL_NAME,
            MODEL_DIR,
            CHECKPOINT_DIR,
            OUTPUT_DIR,
            SEED,
            N_SPLITS,
            MAX_INPUT_LENGTH,
            MAX_INPUT_LENGTH_VAL,
            USE_SAMPLE,
            FREEZE_EMBEDDINGS,
            FREEZE_LAYERS,
            OPTIONS,
            INDICES,
            STOP_WORDS,
        ],
        "params": [TRAINING_ARGS, MODEL_FREEZING, DATA_CONFIG, TOKENIZATION_CONFIG, CV_CONFIG, EARLY_STOPPING],
        "utils": [
            create_option_mappings,
            preprocess_wrapper,
            DataCollatorForMultipleChoice,
            setup_device,
            set_random_seed,
        ],
        "metrics": [competition_score, predictions_to_map_output, compute_metrics, map3_torch, calculate_accuracy_at_k],
        "data_processing": [
            load_training_data,
            prepare_cross_validation_data,
            prepare_final_training_data,
            create_datasets,
            add_combined_text_column,
            get_removable_columns,
            tokenize_dataset,
        ],
        "model": [
            setup_model_and_tokenizer,
            configure_model_freezing,
            setup_training_arguments,
            setup_trainer,
            save_model_and_tokenizer,
            load_trained_model,
            get_model_info,
        ],
    }
