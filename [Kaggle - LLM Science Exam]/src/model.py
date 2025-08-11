"""
Model Setup and Configuration

This module contains functions for setting up the DeBERTa model, including
model freezing, training configuration, and trainer setup. These functions
provide a complete pipeline for initializing, configuring, and training
the multiple choice question answering model.

Functions:
    setup_model_and_tokenizer: Setup the model and tokenizer for training
    configure_model_freezing: Configure model freezing for training
    setup_training_arguments: Setup training arguments with defaults from params
    setup_trainer: Setup the trainer for model training
    save_model_and_tokenizer: Save the trained model and tokenizer
    load_trained_model: Load a trained model and tokenizer
    get_model_info: Get information about the model architecture and parameters
"""

from transformers import AutoModelForMultipleChoice, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments

from .constants import MODEL_DIR, MODEL_NAME
from .params import EARLY_STOPPING, MODEL_FREEZING, TRAINING_ARGS
from .utils import DataCollatorForMultipleChoice


def setup_model_and_tokenizer(model_name: str = None):
    """
    Setup the model and tokenizer for training.
    
    This function loads the pre-trained DeBERTa model and tokenizer from
    Hugging Face. It's the first step in setting up the training pipeline
    and ensures compatibility between the model and tokenizer.

    Args:
        model_name: Name of the model to use (defaults to MODEL_NAME constant)
                    Should be a valid Hugging Face model identifier

    Returns:
        tuple: A tuple containing:
            - tokenizer: Configured tokenizer for the model
            - model: Pre-trained model ready for fine-tuning
            
    Example:
        >>> tokenizer, model = setup_model_and_tokenizer()
        >>> print(f"Model: {type(model)}")
        >>> print(f"Tokenizer: {type(tokenizer)}")
        
        >>> # Use custom model
        >>> tokenizer, model = setup_model_and_tokenizer("microsoft/deberta-base")
    """
    if model_name is None:
        model_name = MODEL_NAME

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMultipleChoice.from_pretrained(model_name)

    return tokenizer, model


def configure_model_freezing(model, freeze_embeddings: bool = None, freeze_layers: int = None):
    """
    Configure model freezing for training.
    
    This function selectively freezes parts of the model to reduce training
    time and memory usage while potentially improving performance by
    preserving pre-trained representations. It can freeze the embedding
    layer and a specified number of encoder layers.

    Args:
        model: The model to configure (should be a DeBERTa model)
        freeze_embeddings: Whether to freeze embeddings (defaults to MODEL_FREEZING["freeze_embeddings"])
        freeze_layers: Number of layers to freeze (defaults to MODEL_FREEZING["freeze_layers"])

    Returns:
        model: Configured model with frozen parameters
        
    Note:
        Freezing embeddings can significantly speed up training and reduce
        memory usage. Freezing early layers preserves more of the pre-trained
        knowledge while allowing later layers to adapt to the specific task.
        
    Example:
        >>> # Freeze embeddings and first 12 layers
        >>> model = configure_model_freezing(model, freeze_embeddings=True, freeze_layers=12)
        >>> 
        >>> # Use default configuration from params
        >>> model = configure_model_freezing(model)
    """
    if freeze_embeddings is None:
        freeze_embeddings = MODEL_FREEZING["freeze_embeddings"]
    if freeze_layers is None:
        freeze_layers = MODEL_FREEZING["freeze_layers"]

    # Freeze embeddings if specified
    if freeze_embeddings:
        print("Freezing embeddings.")
        for param in model.deberta.embeddings.parameters():
            param.requires_grad = False

    # Freeze specified number of layers
    if freeze_layers > 0:
        print(f"Freezing {freeze_layers} layers.")
        for layer in model.deberta.encoder.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    return model


def setup_training_arguments(**kwargs):
    """
    Setup training arguments with defaults from params.
    
    This function creates TrainingArguments for the Hugging Face Trainer,
    using default values from the params module. Any provided keyword
    arguments will override the defaults, allowing for easy customization.

    Args:
        **kwargs: Override default training arguments
                 Common overrides: learning_rate, batch_size, num_epochs

    Returns:
        TrainingArguments: Configured training arguments object
        
    Note:
        The function starts with TRAINING_ARGS from params.py and then
        applies any overrides. This ensures consistency while allowing
        flexibility for different training runs.
        
    Example:
        >>> # Use default arguments
        >>> training_args = setup_training_arguments()
        >>> 
        >>> # Override specific parameters
        >>> training_args = setup_training_arguments(
        ...     learning_rate=1e-5,
        ...     per_device_train_batch_size=8,
        ...     num_train_epochs=10
        ... )
    """
    # Start with default arguments
    args_dict = TRAINING_ARGS.copy()

    # Override with provided arguments
    args_dict.update(kwargs)

    return TrainingArguments(**args_dict)


def setup_trainer(model, tokenizer, train_dataset, eval_dataset, training_args=None, compute_metrics=None):
    """
    Setup the trainer for model training.
    
    This function creates a complete Trainer instance with all necessary
    components including the model, datasets, data collator, and callbacks.
    It's the main function for setting up the training pipeline.

    Args:
        model: The model to train (should be configured with freezing if desired)
        tokenizer: The tokenizer to use (should match the model)
        train_dataset: Training dataset (Hugging Face Dataset)
        eval_dataset: Evaluation dataset (Hugging Face Dataset)
        training_args: Training arguments (optional, will use defaults if None)
        compute_metrics: Function to compute metrics (optional, will use default if None)

    Returns:
        Trainer: Configured trainer instance ready for training
        
    Note:
        The function automatically sets up:
        - Data collator for multiple choice tasks
        - Early stopping callback with parameters from EARLY_STOPPING
        - Default compute_metrics function if none provided
        
    Example:
        >>> trainer = setup_trainer(
        ...     model, tokenizer, train_dataset, eval_dataset,
        ...     training_args=training_args
        ... )
        >>> trainer.train()
    """
    if training_args is None:
        training_args = setup_training_arguments()

    if compute_metrics is None:
        from .metrics import compute_metrics

    # Setup data collator
    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)

    # Setup early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=EARLY_STOPPING["early_stopping_patience"],
        early_stopping_threshold=EARLY_STOPPING["early_stopping_threshold"],
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )

    return trainer


def save_model_and_tokenizer(model, tokenizer, model_dir: str = None):
    """
    Save the trained model and tokenizer.
    
    This function saves both the fine-tuned model and its tokenizer to
    the specified directory. The saved model can be loaded later for
    inference or further fine-tuning.

    Args:
        model: The trained model to save
        tokenizer: The tokenizer to save
        model_dir: Directory to save the model (defaults to MODEL_DIR constant)

    Returns:
        str: Path where the model was saved
        
    Note:
        The function saves both the model weights and the tokenizer
        configuration, ensuring that the model can be loaded correctly
        for inference.
        
    Example:
        >>> save_path = save_model_and_tokenizer(model, tokenizer)
        >>> print(f"Model saved to: {save_path}")
        >>> 
        >>> # Save to custom directory
        >>> save_path = save_model_and_tokenizer(model, tokenizer, "my_model")
    """
    if model_dir is None:
        model_dir = MODEL_DIR

    # Save model and tokenizer
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    print(f"Model and tokenizer saved to {model_dir}")
    return model_dir


def load_trained_model(model_dir: str = None):
    """
    Load a trained model and tokenizer.
    
    This function loads a previously saved model and tokenizer from
    the specified directory. It's useful for loading models for
    inference or continuing training from a checkpoint.

    Args:
        model_dir: Directory containing the saved model (defaults to MODEL_DIR constant)

    Returns:
        tuple: A tuple containing:
            - tokenizer: Loaded tokenizer
            - model: Loaded model ready for inference or further training
            
    Example:
        >>> tokenizer, model = load_trained_model()
        >>> 
        >>> # Load from specific directory
        >>> tokenizer, model = load_trained_model("checkpoints/epoch_5")
    """
    if model_dir is None:
        model_dir = MODEL_DIR

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForMultipleChoice.from_pretrained(model_dir)

    return tokenizer, model


def get_model_info(model):
    """
    Get information about the model architecture and parameters.
    
    This function analyzes the model to provide information about the
    total number of parameters, how many are trainable vs. frozen,
    and the percentage of frozen parameters. It's useful for
    understanding the model's training configuration.

    Args:
        model: The model to analyze (should be a PyTorch model)

    Returns:
        dict: Dictionary containing model information:
            - total_parameters: Total number of parameters
            - trainable_parameters: Number of trainable parameters
            - frozen_parameters: Number of frozen parameters
            - frozen_percentage: Percentage of parameters that are frozen
            
    Example:
        >>> info = get_model_info(model)
        >>> print(f"Total parameters: {info['total_parameters']:,}")
        >>> print(f"Trainable: {info['trainable_parameters']:,}")
        >>> print(f"Frozen: {info['frozen_parameters']:,} ({info['frozen_percentage']:.1f}%)")
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "frozen_percentage": (frozen_params / total_params) * 100,
    }
