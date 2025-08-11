"""
Utility Functions and Classes

This module contains utility functions and classes used throughout the LLM Science Exam
training pipeline, including preprocessing functions, data collators, and device setup.

Functions:
    create_option_mappings: Create mappings between answer options and indices
    preprocess_wrapper: Create preprocessing functions for tokenization
    setup_device: Setup and return the appropriate device (GPU/CPU)
    set_random_seed: Set random seed for reproducibility

Classes:
    DataCollatorForMultipleChoice: Data collator for multiple choice tasks
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase


def create_option_mappings():
    """
    Create option to index and index to option mappings for answer choices.

    This function creates two dictionaries that map between the letter options
    (A, B, C, D, E) and their corresponding numerical indices (0, 1, 2, 3, 4).
    This is useful for converting between human-readable answer choices and
    model output indices.

    Returns:
        tuple: A tuple containing two dictionaries:
            - option_to_index: Maps letters to indices (e.g., 'A' -> 0)
            - index_to_option: Maps indices to letters (e.g., 0 -> 'A')

    Example:
        >>> option_to_index, index_to_option = create_option_mappings()
        >>> option_to_index['A']
        0
        >>> index_to_option[0]
        'A'
    """
    from .constants import INDICES, OPTIONS

    option_to_index = {option: index for option, index in zip(OPTIONS, INDICES)}
    index_to_option = {index: option for option, index in zip(OPTIONS, INDICES)}

    return option_to_index, index_to_option


def preprocess_wrapper(tokenizer: PreTrainedTokenizer, max_input: int = 350):
    """
    Create a preprocessing function for tokenizing questions and answers.

    This function returns a preprocessing function that can be used with the
    Hugging Face datasets library. The returned function handles both regular
    questions and questions with context, tokenizing the input and preparing
    labels for multiple choice training.

    Args:
        tokenizer: The tokenizer to use for text processing
        max_input: Maximum input length for tokenization (default: 350)

    Returns:
        function: A preprocessing function that can be applied to dataset examples

    Note:
        The returned function automatically detects whether context is present
        and applies the appropriate preprocessing strategy.

    Example:
        >>> preprocess_func = preprocess_wrapper(tokenizer, max_input=450)
        >>> tokenized_dataset = dataset.map(preprocess_func, batched=False)
    """
    from .constants import OPTIONS

    def preprocess(example):
        """
        Preprocess a single example from the dataset.

        This function handles examples without context, combining the question
        prompt with each answer option for tokenization.

        Args:
            example: Dictionary containing prompt, options, and answer
                Required keys: 'prompt', 'A', 'B', 'C', 'D', 'E', 'answer'

        Returns:
            dict: Tokenized example with label, ready for model training
        """
        option_to_index, _ = create_option_mappings()

        first_sentence = [example["prompt"]] * 5
        second_sentence = [example[option] for option in OPTIONS]

        tokenized_example = tokenizer(first_sentence, second_sentence, truncation=True)

        tokenized_example["label"] = option_to_index[example["answer"]]
        return tokenized_example

    def preprocess_with_context(example):
        """
        Preprocess a single example with context.

        This function handles examples that include context information,
        combining context, question, and answer options in a structured format.

        Args:
            example: Dictionary containing context, prompt, options, and answer
                Required keys: 'context', 'prompt', 'A', 'B', 'C', 'D', 'E', 'answer'

        Returns:
            dict: Tokenized example with label, ready for model training
        """
        option_to_index, _ = create_option_mappings()

        first_sentence = ["[CLS] " + example["context"]] * 5
        second_sentences = [" #### " + example["prompt"] + " [SEP] " + example[option] + " [SEP]" for option in OPTIONS]

        tokenized_example = tokenizer(
            first_sentence,
            second_sentences,
            truncation="only_first",
            max_length=max_input,
            add_special_tokens=False,
        )
        tokenized_example["label"] = option_to_index[example["answer"]]

        return tokenized_example

    return preprocess_with_context


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator for multiple choice tasks.

    This class handles formatting and batching data for multiple-choice tasks,
    ensuring proper padding and tensor conversion. It's designed to work with
    the Hugging Face Trainer class and handles the specific requirements of
    multiple choice question answering.

    Attributes:
        tokenizer: The tokenizer used for padding and tensor conversion
        padding: Padding strategy (True, False, or specific strategy)
        max_length: Maximum sequence length for padding
        pad_to_multiple_of: Pad sequences to multiples of this value

    Methods:
        __call__: Process a batch of features and return formatted tensors
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        """
        Process a batch of features for multiple choice training.

        This method takes a list of feature dictionaries (each representing
        a question with multiple answer choices) and converts them into a
        properly formatted batch for the model.

        Args:
            features: List of feature dictionaries, where each dictionary
                contains tokenized input for a question with multiple choices

        Returns:
            dict: A batch dictionary containing:
                - input_ids: Padded input token IDs
                - attention_mask: Attention mask for the inputs
                - labels: Answer labels as tensors

        Note:
            The method automatically handles the restructuring of features
            from question-choice pairs to individual examples, applies
            proper padding, and reshapes the output for the model.
        """
        # Find the correct label key
        label_name = "label" if "label" in features[0].keys() else "labels"

        # Extract labels and remove them from features
        labels = [feature.pop(label_name) for feature in features]

        # Get batch size and number of choices
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        # Restructure features so each question-choice pair becomes a separate example
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        # Pad all sequences to the same length
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Reshape the batch back into the original format
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}

        # Add the labels back into the batch as a tensor
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)

        return batch


def setup_device():
    """
    Setup and return the appropriate device (GPU/CPU) for training.

    This function automatically detects whether CUDA is available and
    sets up the appropriate device for PyTorch operations. It prints
    a message indicating which device is being used.

    Returns:
        torch.device: The device to use for training (cuda or cpu)

    Example:
        >>> device = setup_device()
        GPU is available
        >>> print(device)
        device(type='cuda')
    """
    if torch.cuda.is_available():
        print("GPU is available")
        device = torch.device("cuda")
    else:
        print("GPU is not available")
        device = torch.device("cpu")

    return device


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all random number generators.

    This function sets the random seed for Python's random module, NumPy,
    and PyTorch (including CUDA if available) to ensure reproducible
    results across different runs.

    Args:
        seed: Random seed value (default: 42)

    Example:
        >>> set_random_seed(123)
        >>> # All subsequent random operations will be reproducible
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
