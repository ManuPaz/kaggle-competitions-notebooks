"""
Model configurations for inference in the Child Mind Institute - Detect Sleep States project
"""

from .params import NUM_EPOCHS, STEPS_PER_EPOCH

# Base configuration
CFG = {
    "num_epochs": NUM_EPOCHS,
    "steps_per_epoch": STEPS_PER_EPOCH,
    "patch_size": 12,
    "block_size": 17280,
    "stride": 17280,
    "model_dim": 320,
    "time_mixing_dim": 1440,
    "feature_mixing_dim": 320,
    "model_num_heads": 6,
    "model_num_encoder_layers": 5,
    "model_num_lstm_layers": 5,
    "model_first_dropout": 0.5,
    "model_second_dropout": 0.5,
    "model_encoder_dropout": 0.1,
    "model_mha_dropout": 0.0,
}

# LSTM with dropout configuration
CFG_lstm_dropout = {
    "num_epochs": NUM_EPOCHS,
    "steps_per_epoch": STEPS_PER_EPOCH,
    "patch_size": 10,
    "block_size": 17280,
    "stride": 17280,
    "model_dim": 160,
    "time_mixing_dim": 1440,
    "feature_mixing_dim": 160,
    "model_num_heads": 6,
    "model_num_encoder_layers": 5,
    "model_num_lstm_layers": 5,
    "model_first_dropout": 0.5,
    "model_second_dropout": 0.5,
    "model_encoder_dropout": 0.1,
    "model_mha_dropout": 0.0,
}

# Attention configuration
CFG_Att = {
    "num_epochs": NUM_EPOCHS,
    "steps_per_epoch": STEPS_PER_EPOCH,
    "patch_size": 10,
    "block_size": 17280,
    "stride": 17280,
    "model_dim": 160,
    "time_mixing_dim": 1440,
    "feature_mixing_dim": 80,
    "model_num_heads": 6,
    "model_num_encoder_layers": 5,
    "model_num_lstm_layers": 2,
    "model_first_dropout": 0.5,
    "model_second_dropout": 0.5,
    "model_encoder_dropout": 0.1,
    "model_mha_dropout": 0.0,
    "model_transformer_layers": 3,
    "transformer_dim": 320,
}

# CNN configuration
CFG_CNN = {
    "num_epochs": NUM_EPOCHS,
    "steps_per_epoch": STEPS_PER_EPOCH,
    "patch_size": 10,
    "block_size": 17280,
    "stride": 17280,
    "model_dim": 160,
    "time_mixing_dim": 1440,
    "feature_mixing_dim": 160,
    "model_num_heads": 6,
    "model_num_encoder_layers": 5,
    "model_num_lstm_layers": 2,
    "model_cnn_layers": 3,
    "model_first_dropout": 0.5,
    "model_second_dropout": 0.5,
    "model_encoder_dropout": 0.1,
    "model_mha_dropout": 0.0,
}

# Conv1 configuration
CFGConv1 = {
    "num_epochs": NUM_EPOCHS,
    "steps_per_epoch": STEPS_PER_EPOCH,
    "patch_size": 1,
    "block_size": 17280,
    "stride": 17280,
    "model_dim": 160,
    "time_mixing_dim": 1440,
    "feature_mixing_dim": 160,
    "model_num_heads": 6,
    "model_num_encoder_layers": 3,
    "model_num_lstm_layers": 5,
    "model_first_dropout": 0.2,
    "model_second_dropout": 0.5,
    "model_encoder_dropout": 0.1,
    "model_mha_dropout": 0.0,
}

# Conv1 with LSTM dropout configuration
CFG_CONV1_LSTMDROPOUT = {
    "num_epochs": NUM_EPOCHS,
    "steps_per_epoch": STEPS_PER_EPOCH,
    "patch_size": 1,
    "block_size": 17280,
    "stride": 17280,
    "model_dim": 160,
    "time_mixing_dim": 1440,
    "feature_mixing_dim": 160,
    "model_num_heads": 6,
    "model_num_encoder_layers": 3,
    "model_num_lstm_layers": 5,
    "model_first_dropout": 0.2,
    "model_second_dropout": 0.5,
    "model_encoder_dropout": 0.1,
    "model_mha_dropout": 0.0,
}
