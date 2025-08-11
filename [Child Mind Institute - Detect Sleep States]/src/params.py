"""
Model parameters and configuration for the Child Mind Institute - Detect Sleep States project
"""

# Training parameters
LEARNING_RATE = 0.0005
STEPS_PER_EPOCH = 500
NUM_EPOCHS = 14
WARMUP_STEPS = 300
GPU_BATCH_SIZE = 32

# Optimizer parameters
OPTMIZER_BETA1 = 0.9
OPTMIZER_BETA2 = 0.98
OPTMIZER_EPSILON = 1e-9

# Training flags
ONLY_TEST = False
FOLD = 1
SAMPLE_NORMALIZE = True
DROP_INITIAL_DATE = True
TRAIN = True
PREDICT = True

# Model configuration
CFG = {
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
    "model_first_dropout": 0.5,
    "model_second_dropout": 0.5,
    "model_encoder_dropout": 0.1,
    "model_mha_dropout": 0.0,
}

# Derived dimensions
DIM = CFG["patch_size"] * 2 + 2  # 2 numeric features * patch size + 2 more features (sine and cosine of day time)
TAM_CONSIDER = CFG["patch_size"]
