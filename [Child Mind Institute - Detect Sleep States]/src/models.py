"""
TensorFlow models for the Child Mind Institute - Detect Sleep States project
"""

import tensorflow as tf

from .params import CFG


class Encoder(tf.keras.Model):
    """
    Encoder layer for the neural network model
    """

    def __init__(self):
        super().__init__()

        self.first_linear = tf.keras.layers.Dense(CFG["time_mixing_dim"])
        self.second_linear = tf.keras.layers.Dense(CFG["feature_mixing_dim"])

        self.add = tf.keras.layers.Add()

        self.first_dropout = tf.keras.layers.Dropout(CFG["model_first_dropout"])
        self.second_dropout = tf.keras.layers.Dropout(CFG["model_second_dropout"])

    def call(self, x, training=None):
        """
        Forward pass through the encoder

        Args:
            x: Input tensor
            training: Training flag

        Returns:
            Processed tensor
        """
        features_mixing = self.second_linear(x)
        features_mixing = tf.keras.layers.ReLU()(features_mixing)
        features_mixing = self.second_dropout(features_mixing)

        x = self.add([x, features_mixing])
        return x


class Model(tf.keras.Model):
    """
    Main neural network model with encoders and LSTM layers
    """

    def __init__(self):
        super().__init__()

        self.encoders = [Encoder() for i in range(CFG["model_num_encoder_layers"])]

        self.lstm_layers = [
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(CFG["model_dim"], return_sequences=True))
            for _ in range(CFG["model_num_lstm_layers"])
        ]

        self.first_linear = tf.keras.layers.Dense(CFG["feature_mixing_dim"])
        self.first_dropout = tf.keras.layers.Dropout(0.1)
        self.last_linear = tf.keras.layers.Dense(2)

    def call(self, x):
        """
        Forward pass through the model

        Args:
            x: Input tensor

        Returns:
            Output tensor with sigmoid activation
        """
        x = self.first_linear(x)
        x = self.first_dropout(x)

        for i in range(CFG["model_num_encoder_layers"]):
            x = self.encoders[i](x)

        for i in range(CFG["model_num_lstm_layers"]):
            x = self.lstm_layers[i](x)

        x = self.last_linear(x)
        x = tf.nn.sigmoid(x)
        return x
