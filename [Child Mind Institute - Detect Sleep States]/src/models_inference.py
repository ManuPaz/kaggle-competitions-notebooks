"""
Model classes for inference in the Child Mind Institute - Detect Sleep States project
"""

import tensorflow as tf
from .configs_inference import CFGConv1, CFG_CONV1_LSTMDROPOUT


class EncoderConv1(tf.keras.Model):
    """
    Convolutional encoder for Conv1 models
    """
    
    def __init__(self, j):
        super().__init__()
        
        self.first_linear = tf.keras.layers.Dense(CFGConv1["time_mixing_dim"])
        self.second_linear = tf.keras.layers.Dense(CFGConv1["feature_mixing_dim"])
        
        self.conv1 = tf.keras.layers.Conv1D(32 * (2**j), 3, strides=1, padding="same")
        self.conv2 = tf.keras.layers.Conv1D(32 * (2**j), 3, strides=1, padding="same")
        self.conv3 = tf.keras.layers.Conv1D(32 * (2**j), 3, strides=1, padding="same")
        
        self.max_pool = tf.keras.layers.AveragePooling1D(pool_size=2)
        self.dropout = tf.keras.layers.Dropout(0.2)
    
    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.max_pool(x)
        return x


class ModelConv1(tf.keras.Model):
    """
    Conv1 model with convolutional encoders and LSTM layers
    """
    
    def __init__(self):
        super().__init__()
        
        self.encoders = [
            EncoderConv1(i + 1) for i in range(CFGConv1["model_num_encoder_layers"])
        ]
        self.lstm_layers = [
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(CFGConv1["model_dim"], return_sequences=True)
            )
            for _ in range(CFGConv1["model_num_lstm_layers"])
        ]
        
        self.linear = tf.keras.layers.Dense(64)
        self.last_linear = tf.keras.layers.Dense(2)
    
    def call(self, x):
        for i in range(CFGConv1["model_num_encoder_layers"]):
            x = self.encoders[i](x)
        
        for i in range(CFGConv1["model_num_lstm_layers"]):
            x = self.lstm_layers[i](x)
        
        x = self.linear(x)
        x = self.last_linear(x)
        x = tf.nn.sigmoid(x)
        return x


class EncoderConv6(tf.keras.Model):
    """
    Convolutional encoder for Conv6 models
    """
    
    def __init__(self, j):
        super().__init__()
        self.j = j
        
        self.first_linear = tf.keras.layers.Dense(CFGConv1["time_mixing_dim"])
        self.second_linear = tf.keras.layers.Dense(CFGConv1["feature_mixing_dim"])
        
        self.conv1 = tf.keras.layers.Conv1D(32 * (2**j), 3, strides=1, padding="same")
        self.conv2 = tf.keras.layers.Conv1D(32 * (2**j), 3, strides=1, padding="same")
        self.conv3 = tf.keras.layers.Conv1D(32 * (2**j), 3, strides=1, padding="same")
        
        if self.j == 1:
            POOL = 2
        else:
            POOL = 3
        self.max_pool = tf.keras.layers.AveragePooling1D(pool_size=POOL)
        self.dropout = tf.keras.layers.Dropout(0.2)
    
    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = tf.keras.layers.ReLU()(x)
        
        if self.j < 3:
            x = self.max_pool(x)
        
        return x


class ModelConv6(tf.keras.Model):
    """
    Conv6 model with convolutional encoders and LSTM layers
    """
    
    def __init__(self):
        super().__init__()
        
        self.encoders = [
            EncoderConv6(i + 1) for i in range(CFGConv1["model_num_encoder_layers"])
        ]
        self.lstm_layers = [
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(CFGConv1["model_dim"], return_sequences=True)
            )
            for _ in range(CFGConv1["model_num_lstm_layers"])
        ]
        
        self.linear = tf.keras.layers.Dense(64)
        self.last_linear = tf.keras.layers.Dense(2)
    
    def call(self, x):
        for i in range(CFGConv1["model_num_encoder_layers"]):
            x = self.encoders[i](x)
        
        for i in range(CFGConv1["model_num_lstm_layers"]):
            x = self.lstm_layers[i](x)
        
        x = self.linear(x)
        x = self.last_linear(x)
        x = tf.nn.sigmoid(x)
        return x


class EncoderConv5(tf.keras.Model):
    """
    Convolutional encoder for Conv5 models
    """
    
    def __init__(self, j):
        super().__init__()
        self.j = j
        
        self.first_linear = tf.keras.layers.Dense(CFGConv1["time_mixing_dim"])
        self.second_linear = tf.keras.layers.Dense(CFGConv1["feature_mixing_dim"])
        
        self.conv1 = tf.keras.layers.Conv1D(32 * (2**j), 3, strides=1, padding="same")
        self.conv2 = tf.keras.layers.Conv1D(32 * (2**j), 3, strides=1, padding="same")
        self.conv3 = tf.keras.layers.Conv1D(32 * (2**j), 3, strides=1, padding="same")
        
        self.max_pool = tf.keras.layers.AveragePooling1D(pool_size=2)
        self.dropout = tf.keras.layers.Dropout(0.2)
    
    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = tf.keras.layers.ReLU()(x)
        
        if self.j < 3:
            x = self.max_pool(x)
        
        return x


class ModelConv5(tf.keras.Model):
    """
    Conv5 model with convolutional encoders and LSTM layers
    """
    
    def __init__(self):
        super().__init__()
        
        self.encoders = [
            EncoderConv5(i + 1) for i in range(CFGConv1["model_num_encoder_layers"])
        ]
        self.lstm_layers = [
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(CFGConv1["model_dim"], return_sequences=True)
            )
            for _ in range(CFGConv1["model_num_lstm_layers"])
        ]
        
        self.linear = tf.keras.layers.Dense(64)
        self.last_linear = tf.keras.layers.Dense(2)
    
    def call(self, x):
        for i in range(CFGConv1["model_num_encoder_layers"]):
            x = self.encoders[i](x)
        
        for i in range(CFGConv1["model_num_lstm_layers"]):
            x = self.lstm_layers[i](x)
        
        x = self.linear(x)
        x = self.last_linear(x)
        x = tf.nn.sigmoid(x)
        return x


class EncoderConv1_LSTMDROPOUT(tf.keras.Model):
    """
    Convolutional encoder for Conv1 LSTM dropout models
    """
    
    def __init__(self, j):
        super().__init__()
        
        self.first_linear = tf.keras.layers.Dense(CFG_CONV1_LSTMDROPOUT["time_mixing_dim"])
        self.second_linear = tf.keras.layers.Dense(CFG_CONV1_LSTMDROPOUT["feature_mixing_dim"])
        
        self.conv1 = tf.keras.layers.Conv1D(32 * (2**j), 3, strides=1, padding="same")
        self.conv2 = tf.keras.layers.Conv1D(32 * (2**j), 3, strides=1, padding="same")
        self.conv3 = tf.keras.layers.Conv1D(32 * (2**j), 3, strides=1, padding="same")
        
        self.max_pool = tf.keras.layers.AveragePooling1D(pool_size=2)
        self.dropout = tf.keras.layers.Dropout(0.2)
    
    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.max_pool(x)
        return x


class ModelConv1_LSTMDROPOUT(tf.keras.Model):
    """
    Conv1 LSTM dropout model with convolutional encoders and LSTM layers with dropout
    """
    
    def __init__(self):
        super().__init__()
        
        self.encoders = [
            EncoderConv1_LSTMDROPOUT(i + 1)
            for i in range(CFG_CONV1_LSTMDROPOUT["model_num_encoder_layers"])
        ]
        self.lstm_layers = [
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    CFG_CONV1_LSTMDROPOUT["model_dim"],
                    return_sequences=True,
                    dropout=0.2,
                )
            )
            for _ in range(CFG_CONV1_LSTMDROPOUT["model_num_lstm_layers"])
        ]
        
        self.linear = tf.keras.layers.Dense(64)
        self.last_linear = tf.keras.layers.Dense(2)
    
    def call(self, x):
        for i in range(CFG_CONV1_LSTMDROPOUT["model_num_encoder_layers"]):
            x = self.encoders[i](x)
        
        for i in range(CFG_CONV1_LSTMDROPOUT["model_num_lstm_layers"]):
            x = self.lstm_layers[i](x)
        
        x = self.linear(x)
        x = self.last_linear(x)
        x = tf.nn.sigmoid(x)
        return x


class Encoder(tf.keras.Model):
    """
    Standard encoder with feature mixing
    """
    
    def __init__(self, CFG):
        super().__init__()
        
        self.first_linear = tf.keras.layers.Dense(CFG["time_mixing_dim"])
        self.second_linear = tf.keras.layers.Dense(CFG["feature_mixing_dim"])
        
        self.add = tf.keras.layers.Add()
        self.first_dropout = tf.keras.layers.Dropout(CFG["model_first_dropout"])
        self.second_dropout = tf.keras.layers.Dropout(CFG["model_second_dropout"])
    
    def call(self, x, training=None):
        features_mixing = self.second_linear(x)
        features_mixing = tf.keras.layers.ReLU()(features_mixing)
        features_mixing = self.second_dropout(features_mixing)
        
        x = self.add([x, features_mixing])
        return x


class Model(tf.keras.Model):
    """
    Standard model with encoders and LSTM layers
    """
    
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG
        
        self.encoders = [
            Encoder(self.CFG) for i in range(self.CFG["model_num_encoder_layers"])
        ]
        
        self.lstm_layers = [
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(self.CFG["model_dim"], return_sequences=True)
            )
            for _ in range(self.CFG["model_num_lstm_layers"])
        ]
        
        self.first_linear = tf.keras.layers.Dense(self.CFG["feature_mixing_dim"])
        self.first_dropout = tf.keras.layers.Dropout(0.1)
        self.last_linear = tf.keras.layers.Dense(2)
    
    def call(self, x):
        x = self.first_linear(x)
        x = self.first_dropout(x)
        
        for i in range(self.CFG["model_num_encoder_layers"]):
            x = self.encoders[i](x)
        
        for i in range(self.CFG["model_num_lstm_layers"]):
            x = self.lstm_layers[i](x)
        
        x = self.last_linear(x)
        x = tf.nn.sigmoid(x)
        return x


class EncoderGRU(tf.keras.Model):
    """
    Encoder for GRU models
    """
    
    def __init__(self, CFG):
        super().__init__()
        
        self.first_linear = tf.keras.layers.Dense(CFG["time_mixing_dim"])
        self.second_linear = tf.keras.layers.Dense(CFG["feature_mixing_dim"])
        
        self.add = tf.keras.layers.Add()
        self.first_dropout = tf.keras.layers.Dropout(CFG["model_first_dropout"])
        self.second_dropout = tf.keras.layers.Dropout(CFG["model_second_dropout"])
    
    def call(self, x, training=None):
        features_mixing = self.second_linear(x)
        features_mixing = tf.keras.layers.ReLU()(features_mixing)
        features_mixing = self.second_dropout(features_mixing)
        
        x = self.add([x, features_mixing])
        return x


class ModelGRU(tf.keras.Model):
    """
    GRU model with encoders and GRU layers
    """
    
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG
        
        self.encoders = [
            EncoderGRU(CFG) for i in range(CFG["model_num_encoder_layers"])
        ]
        
        self.lstm_layers = [
            tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(CFG["model_dim"], return_sequences=True)
            )
            for _ in range(CFG["model_num_lstm_layers"])
        ]
        
        self.first_linear = tf.keras.layers.Dense(CFG["feature_mixing_dim"])
        self.first_dropout = tf.keras.layers.Dropout(0.1)
        self.last_linear = tf.keras.layers.Dense(2)
    
    def call(self, x):
        x = self.first_linear(x)
        x = self.first_dropout(x)
        
        for i in range(self.CFG["model_num_encoder_layers"]):
            x = self.encoders[i](x)
        
        for i in range(self.CFG["model_num_lstm_layers"]):
            x = self.lstm_layers[i](x)
        
        x = self.last_linear(x)
        x = tf.nn.sigmoid(x)
        return x


class Encoder_lstm_dropout(tf.keras.Model):
    """
    Encoder for LSTM dropout models
    """
    
    def __init__(self):
        super().__init__()
        
        from .configs_inference import CFG_lstm_dropout
        
        self.first_linear = tf.keras.layers.Dense(CFG_lstm_dropout["time_mixing_dim"])
        self.second_linear = tf.keras.layers.Dense(CFG_lstm_dropout["feature_mixing_dim"])
        
        self.add = tf.keras.layers.Add()
        self.first_dropout = tf.keras.layers.Dropout(CFG_lstm_dropout["model_first_dropout"])
        self.second_dropout = tf.keras.layers.Dropout(CFG_lstm_dropout["model_second_dropout"])
    
    def call(self, x, training=None):
        features_mixing = self.second_linear(x)
        features_mixing = tf.keras.layers.ReLU()(features_mixing)
        features_mixing = self.second_dropout(features_mixing)
        
        x = self.add([x, features_mixing])
        return x


class Model_lstm_dropout(tf.keras.Model):
    """
    LSTM dropout model with encoders and LSTM layers with dropout
    """
    
    def __init__(self):
        super().__init__()
        
        from .configs_inference import CFG_lstm_dropout
        
        self.encoders = [
            Encoder_lstm_dropout()
            for i in range(CFG_lstm_dropout["model_num_encoder_layers"])
        ]
        
        self.lstm_layers = [
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    CFG_lstm_dropout["model_dim"], return_sequences=True, dropout=0.2
                )
            )
            for _ in range(CFG_lstm_dropout["model_num_lstm_layers"])
        ]
        
        self.first_linear = tf.keras.layers.Dense(CFG_lstm_dropout["feature_mixing_dim"])
        self.first_dropout = tf.keras.layers.Dropout(0.1)
        self.last_linear = tf.keras.layers.Dense(2)
    
    def call(self, x):
        x = self.first_linear(x)
        x = self.first_dropout(x)
        
        for i in range(CFG_lstm_dropout["model_num_encoder_layers"]):
            x = self.encoders[i](x)
        
        for i in range(CFG_lstm_dropout["model_num_lstm_layers"]):
            x = self.lstm_layers[i](x)
        
        x = self.last_linear(x)
        x = tf.nn.sigmoid(x)
        return x
