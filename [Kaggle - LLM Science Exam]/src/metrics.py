"""
Evaluation Metrics and Scoring Functions

This module contains all the evaluation metrics and scoring functions used for
assessing the performance of the LLM Science Exam model, including the competition's
Map@3 metric and other accuracy measures.

Functions:
    competition_score: Calculate the competition's Map@3 score
    predictions_to_map_output: Convert model predictions to Map@3 format
    compute_metrics: Compute metrics for Hugging Face Trainer
    map3_torch: PyTorch implementation of Map@3 metric
    calculate_accuracy_at_k: Calculate accuracy at k for predictions
"""

import numpy as np
import torch


def competition_score(y_true, y_pred):
    """
    Calculate the competition score (Map@3) for the model.
    
    This function implements the Map@3 metric used in the competition, which gives
    full credit for the first correct answer, half credit for the second, and
    one-third credit for the third. This metric rewards models that can rank
    correct answers higher in their predictions.

    Args:
        y_true: List of true answer labels (e.g., ['A', 'B', 'C'])
        y_pred: List of predicted answer labels, where each element is a list
                of top 3 predictions (e.g., [['A', 'B', 'C'], ['B', 'A', 'D']])

    Returns:
        float: Map@3 score between 0 and 1, where 1.0 is perfect prediction
        
    Example:
        >>> y_true = ['A', 'B', 'C']
        >>> y_pred = [['A', 'B', 'D'], ['B', 'A', 'C'], ['C', 'A', 'B']]
        >>> score = competition_score(y_true, y_pred)
        >>> print(score)
        1.0  # Perfect predictions in all cases
    """
    ap_at_3 = 0.0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i][0]:
            ap_at_3 += 1
        elif y_true[i] == y_pred[i][1]:
            ap_at_3 += 1 / 2
        elif y_true[i] == y_pred[i][2]:
            ap_at_3 += 1 / 3

    map3 = ap_at_3 / len(y_true)
    return map3


def predictions_to_map_output(predictions):
    """
    Convert model predictions to Map@3 output format.
    
    This function takes raw model logits and converts them to the top 3
    answer choices for each question, which is the format expected by
    the competition evaluation.

    Args:
        predictions: Model output logits as numpy array or tensor
                    Shape: (n_questions, n_options) where n_options is typically 5

    Returns:
        numpy.ndarray: Top 3 answer choices for each question
                      Shape: (n_questions, 3) with values like 'A', 'B', 'C'
                      
    Example:
        >>> logits = np.array([[0.1, 0.8, 0.2, 0.3, 0.4],
        ...                    [0.9, 0.1, 0.2, 0.3, 0.4]])
        >>> top_answers = predictions_to_map_output(logits)
        >>> print(top_answers)
        [['B', 'E', 'D'], ['A', 'E', 'D']]
    """
    from .utils import create_option_mappings

    _, index_to_option = create_option_mappings()

    # Sort indices in descending order
    sorted_answer_indices = np.argsort(-predictions)

    # Take the first three indices for each row
    top_answer_indices = sorted_answer_indices[:, :3]

    # Transform indices to options (0 -> A, 1 -> B, etc.)
    top_answers = np.vectorize(index_to_option.get)(top_answer_indices)

    return top_answers


def compute_metrics(eval_preds):
    """
    Compute evaluation metrics for the model during training.
    
    This function is designed to work with the Hugging Face Trainer class
    and computes the Map@3 metric for evaluation. It automatically handles
    the conversion of model outputs to the appropriate format.

    Args:
        eval_preds: Tuple of (logits, labels) from the model
                    - logits: Model predictions (numpy array or tensor)
                    - labels: True labels as indices (0-4)

    Returns:
        dict: Dictionary containing the Map@3 score
              Format: {"Map@3": float}
              
    Note:
        This function is automatically called by the Trainer during evaluation
        and should not be called manually in most cases.
        
    Example:
        >>> # This is typically called automatically by the Trainer
        >>> metrics = compute_metrics((logits, labels))
        >>> print(metrics)
        {'Map@3': 0.85}
    """
    from .utils import create_option_mappings

    logits, labels = eval_preds
    _, index_to_option = create_option_mappings()

    # Convert predictions to Map@3 format
    y_pred = predictions_to_map_output(logits)

    # Convert labels to option format
    y_true = [index_to_option[label] for label in labels]

    # Calculate and return Map@3 score
    map3_score = competition_score(y_true, y_pred)

    return {"Map@3": np.round(map3_score, 3)}


def map3_torch(predictions, labels):
    """
    PyTorch implementation of Map@3 metric.
    
    This function provides a PyTorch-native implementation of the Map@3 metric,
    which can be more efficient when working with tensors and can be used
    during training for monitoring performance.

    Args:
        predictions: Model predictions as PyTorch tensor
                    Shape: (batch_size, n_options)
        labels: True labels as PyTorch tensor
                Shape: (batch_size,) with values 0-4

    Returns:
        float: Map@3 score as a Python float
        
    Example:
        >>> predictions = torch.tensor([[0.1, 0.8, 0.2, 0.3, 0.4],
        ...                            [0.9, 0.1, 0.2, 0.3, 0.4]])
        >>> labels = torch.tensor([1, 0])
        >>> score = map3_torch(predictions, labels)
        >>> print(score)
        1.0  # Perfect predictions
    """
    hits = (-predictions).argsort() == labels.unsqueeze(1)

    map3 = (hits[:, 0] * 1 + hits[:, 1] * 1 / 2 + hits[:, 2] * 1 / 3).sum().item() / hits.shape[0]

    return map3


def calculate_accuracy_at_k(predictions, labels, k=3):
    """
    Calculate accuracy at k for the predictions.
    
    This function calculates the percentage of questions where the correct
    answer appears in the top-k predictions. It's useful for understanding
    how well the model ranks correct answers.

    Args:
        predictions: Model predictions as numpy array or tensor
                    Shape: (n_questions, n_options)
        labels: True labels as numpy array or tensor
                Shape: (n_questions,) with values 0-4
        k: Number of top predictions to consider (default: 3)

    Returns:
        float: Accuracy at k as a float between 0 and 1
        
    Example:
        >>> predictions = np.array([[0.1, 0.8, 0.2, 0.3, 0.4],
        ...                         [0.9, 0.1, 0.2, 0.3, 0.4]])
        >>> labels = np.array([1, 0])
        >>> acc_at_3 = calculate_accuracy_at_k(predictions, labels, k=3)
        >>> print(acc_at_3)
        1.0  # Both correct answers are in top-3
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Get top k predictions
    top_k_indices = np.argsort(-predictions, axis=1)[:, :k]

    # Check if true label is in top k
    correct = np.any(top_k_indices == labels.reshape(-1, 1), axis=1)

    return correct.mean()
