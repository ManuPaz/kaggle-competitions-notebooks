"""
Loss functions and learning rate schedulers for inference in the Child Mind Institute - Detect Sleep States project
"""

import tensorflow as tf


# Binary crossentropy loss
ce = tf.keras.losses.BinaryCrossentropy(reduction="none")


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
    # Apply mask to not compute loss on NaN targets
    mask = tf.math.logical_not(tf.math.is_nan(real))
    
    y_true = tf.boolean_mask(real, mask)
    y_pred = tf.boolean_mask(output, mask)
    
    tf.debugging.check_numerics(y_true, message="NaNs in 'real'")
    tf.debugging.check_numerics(y_pred, message="NaNs in 'output'")
    
    loss = ce(tf.expand_dims(y_true, axis=-1), tf.expand_dims(y_pred, axis=-1))
    tf.debugging.check_numerics(loss, message="NaNs in 'loss'")
    
    return tf.reduce_mean(loss)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom learning rate scheduler with warmup
    """
    
    def __init__(self, initial_lr, warmup_steps=1):
        super(CustomSchedule, self).__init__()
        
        self.initial_lr = tf.cast(initial_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return tf.math.minimum(
            self.initial_lr, self.initial_lr * (step / self.warmup_steps)
        )
