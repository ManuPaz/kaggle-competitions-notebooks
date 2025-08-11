"""
Custom TensorFlow callbacks for the Child Mind Institute - Detect Sleep States project
"""

import tensorflow as tf


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom learning rate scheduler that decreases LR by a factor each epoch
    """

    def __init__(self, initial_lr, warmup_steps):
        super(CustomSchedule, self).__init__()
        self.initial_lr = tf.Variable(initial_lr, trainable=False, dtype=tf.float32)
        self.decrease_factor = tf.constant(0.95, dtype=tf.float32)  # Reduce by 5% each epoch

    def decrease_learning_rate(self):
        """Decrease learning rate by the decrease factor"""
        if self.initial_lr >= 0.000001:
            self.initial_lr.assign(self.initial_lr * self.decrease_factor)

    def __call__(self, step):
        """Return current learning rate"""
        return self.initial_lr


class ReduceLROnThreshold(tf.keras.callbacks.Callback):
    """
    Callback to reduce learning rate when validation loss reaches a threshold
    """

    def __init__(self, scheduler, threshold=0.055):
        super(ReduceLROnThreshold, self).__init__()
        self.scheduler = scheduler
        self.threshold = threshold
        self.triggered = False

    def on_epoch_end(self, epoch, logs=None):
        """Reduce learning rate at the end of epoch if threshold is met"""
        logs = logs or {}
        loss = logs.get('val_loss')
        print(self.scheduler.initial_lr)

        if not self.triggered and loss is not None and loss <= self.threshold:
            self.triggered = True

        if self.triggered:
            self.scheduler.decrease_learning_rate()


class MetricsCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to compute and save model weights based on validation scores
    """

    def __init__(self, model=None, verbose=0, FOLD=-1):
        if model is not None:
            self.model = model
        self.max_score_all = 0
        self.max_score_test = 0
        self.fold = FOLD
        print(self.fold)

    def prediction(self):
        """
        Make predictions and compute scores
        
        Returns:
            Tuple of (score_all, score_test)
        """
        # This method requires access to global variables from the training loop
        # It will be implemented in the training module
        pass

    def on_epoch_end(self, epoch, logs=None):
        """Save model weights if scores improve"""
        score_all, score_test = self.prediction()

        if score_all >= self.max_score_all:
            print(f"Saving score all: epoch {epoch} because {score_all}>{self.max_score_all}")
            self.model.save_weights(f"model_weights_{epoch}_fold{self.fold}_all")
            self.max_score_all = score_all

        if score_test >= self.max_score_test:
            print(f"Saving just test: epoch {epoch} because {score_test}>{self.max_score_test}")
            self.model.save_weights(f"model_weights_{epoch}_fold{self.fold}_test")
            self.max_score_test = score_test
