"""
Training Parameters and Hyperparameters Configuration

This module contains all the training parameters, hyperparameters, and configuration
settings for the LLM Science Exam model training pipeline. These configurations
are organized into logical groups for easy management and modification.

Configuration Groups:
    MODEL_FREEZING: Settings for freezing model parameters during training
    DATA_CONFIG: Data processing and sampling configuration
    TOKENIZATION_CONFIG: Tokenization and input length settings
    CV_CONFIG: Cross-validation configuration
    EARLY_STOPPING: Early stopping criteria and patience settings
    TRAINING_ARGS: Default training arguments for the Hugging Face Trainer
    MODEL_NAME: Pre-trained model identifier
"""

# Model Freezing Configuration
# Controls which parts of the model are frozen during fine-tuning
MODEL_FREEZING = {
    "freeze_embeddings": True,  # Whether to freeze the embedding layer
    "freeze_layers": 18,  # Number of encoder layers to freeze (0 = no freezing)
}

# Data Processing Configuration
# Controls data loading, sampling, and cross-validation settings
DATA_CONFIG = {
    "use_sample": True,  # Whether to use a sample of data for testing
    "sample_size": 100,  # Size of the sample if use_sample is True
    "n_splits": 5,  # Number of cross-validation splits
    "random_state": 21,  # Random seed for reproducibility
    "shuffle": True,  # Whether to shuffle data during splitting
}

# Tokenization Configuration
# Controls how input text is tokenized and processed
TOKENIZATION_CONFIG = {
    "max_input_length": 450,  # Maximum input length for training data
    "max_input_length_val": 650,  # Maximum input length for validation data
    "truncation": "only_first",  # Truncation strategy for long sequences
    "add_special_tokens": False,  # Whether to add special tokens automatically
}

# Cross-Validation Configuration
# Settings for cross-validation data splitting
CV_CONFIG = {
    "n_splits": 5,  # Number of CV folds
    "random_state": 21,  # Random seed for reproducible splits
    "shuffle": True,  # Whether to shuffle data before splitting
}

# Early Stopping Configuration
# Controls when training should stop early to prevent overfitting
EARLY_STOPPING = {
    "early_stopping_patience": 5,  # Number of epochs to wait before stopping
    "early_stopping_threshold": 0.01,  # Minimum improvement threshold
}

# Training Arguments Configuration
# Default training arguments for the Hugging Face Trainer
TRAINING_ARGS = {
    "output_dir": "finetuned_deberta",  # Output directory for model and logs
    "num_train_epochs": 3,  # Number of training epochs
    "per_device_train_batch_size": 4,  # Training batch size per device
    "per_device_eval_batch_size": 4,  # Evaluation batch size per device
    "warmup_steps": 500,  # Number of warmup steps for learning rate
    "weight_decay": 0.01,  # Weight decay for regularization
    "logging_dir": "./logs",  # Directory for storing logs
    "logging_steps": 10,  # Log every X steps
    "evaluation_strategy": "steps",  # Evaluation strategy (steps, epoch)
    "eval_steps": 500,  # Evaluate every X steps
    "save_strategy": "steps",  # Save strategy (steps, epoch)
    "save_steps": 500,  # Save every X steps
    "save_total_limit": 2,  # Maximum number of checkpoints to save
    "load_best_model_at_end": True,  # Load the best model at the end
    "metric_for_best_model": "Map@3",  # Metric to use for best model selection
    "greater_is_better": True,  # Whether higher metric values are better
    "learning_rate": 2e-5,  # Learning rate for training
    "fp16": True,  # Use mixed precision training
    "report_to": "none",  # Don't report to external services
}

# Pre-trained model identifier
# Hugging Face model name to use for fine-tuning
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_INPUT_INFERENCE = 2500
MAX_INPUT_TRAIN = 640
MAX_INPUT_VAL = 650
INFERENCE_MODEL_PATH1 = ""
INFERENCE_MODEL_PATH2 = ""
INFERENCE_MODEL_PATH3 = ""
