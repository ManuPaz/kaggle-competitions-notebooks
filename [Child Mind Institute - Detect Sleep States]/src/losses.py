"""
Loss functions and metrics for the Child Mind Institute - Detect Sleep States project
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def loss_function(real, output, name="loss_function"):
    """
    Custom loss function with NaN masking

    Args:
        real: Ground truth values
        output: Model predictions
        name: Function name

    Returns:
        Mean loss value
    """
    # Binary crossentropy loss
    ce = tf.keras.losses.BinaryCrossentropy(reduction="none")

    # Apply mask to not compute loss on NaN targets
    mask = tf.math.logical_not(tf.math.is_nan(real))
    y_true = tf.boolean_mask(real, mask)
    y_pred = tf.boolean_mask(output, mask)

    # Check for NaNs in inputs
    tf.debugging.check_numerics(y_true, message="NaNs in 'real'")
    tf.debugging.check_numerics(y_pred, message="NaNs in 'output'")

    # Compute loss
    loss = ce(tf.expand_dims(y_true, axis=-1), tf.expand_dims(y_pred, axis=-1))
    tf.debugging.check_numerics(loss, message="NaNs in 'loss'")

    return tf.reduce_mean(loss)


def metrics(real, preds__):
    """
    Calculate evaluation metrics

    Args:
        real: Ground truth values
        preds__: Model predictions

    Returns:
        None (prints metrics)
    """
    f__ = loss_function(real, preds__)
    umbral = 0.5
    preds_b = (preds__ > umbral).astype(int)

    real__ = real.reshape(-1)
    preds_b__ = preds_b.reshape(-1)

    # Remove NaN values for metric calculation
    mask = ~np.isnan(real__)
    real_clean = real__[mask]
    preds_clean = preds_b__[mask]

    cm = confusion_matrix(real_clean, preds_clean)
    score = f1_score(real_clean, preds_clean)
    accuracy = accuracy_score(real_clean, preds_clean)

    print(f"Loss: {f__}, cm: {cm}, f1-score: {score}, accuracy: {accuracy}")
