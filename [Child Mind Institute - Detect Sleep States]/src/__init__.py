"""
Child Mind Institute - Detect Sleep States project package
"""

from .callbacks import CustomSchedule, MetricsCallback, ReduceLROnThreshold
from .configs_inference import CFG, CFG_CNN, CFG_CONV1_LSTMDROPOUT, CFG_Att, CFG_lstm_dropout, CFGConv1
from .constants import DATA_PATH, INIT, NUMERIC_FEATURES, SIGMA, TARGET, TEST_IDS

# Inference modules
from .constants_inference import (
    ATT,
    CNN,
    CONV1,
    CONV1_LSTMDROPOUT,
    CONV2,
    CONV_5,
    CONV_6,
    DEFAULT_PATCH_SIZE,
    EXTEND,
    GRU,
    LINEAR,
    REPLACE,
    VALIDATE,
    W_BASE,
    W_BASE_5,
    W_BASE_8,
    W_CONV,
    LINEAR_lstm_dropout,
)
from .data_processing import read_data, read_data_test
from .data_processing_inference import (
    drop_initial_date,
    process_df,
    read_data,
    read_data_validate,
    sample_normalize,
    truncate_days,
)
from .distributions import gauss, gauss_standard, lognorm, lognormal_standard, student_t
from .evaluation import build_preds_dict, get_event, get_events, get_preds_df, get_real_event, get_scores
from .losses import loss_function, metrics
from .losses_inference import CustomSchedule, loss_function
from .models import Encoder, Model
from .models_inference import (
    Encoder,
    Encoder_lstm_dropout,
    EncoderConv1,
    EncoderConv1_LSTMDROPOUT,
    EncoderConv5,
    EncoderConv6,
    EncoderGRU,
    Model,
    Model_lstm_dropout,
    ModelConv1,
    ModelConv1_LSTMDROPOUT,
    ModelConv5,
    ModelConv6,
    ModelGRU,
)
from .modelspec_inference import ModelSpec
from .params import (
    CFG,
    DIM,
    DROP_INITIAL_DATE,
    FOLD,
    GPU_BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    ONLY_TEST,
    OPTMIZER_BETA1,
    OPTMIZER_BETA2,
    OPTMIZER_EPSILON,
    PREDICT,
    SAMPLE_NORMALIZE,
    STEPS_PER_EPOCH,
    TAM_CONSIDER,
    TRAIN,
    WARMUP_STEPS,
)
from .prediction_inference import get_df, get_event, get_events, get_preds_df, predict_model
from .training import (
    create_datasets,
    get_inds,
    prepare_validation_data,
    read_indices,
    read_val_indices,
    reshape_features,
)
from .utils import drop_initial_date, sample_normalize, truncate_days

__all__ = [
    # Constants
    "TARGET",
    "TEST_IDS",
    "SIGMA",
    "NUMERIC_FEATURES",
    "DATA_PATH",
    "INIT",
    # Parameters
    "LEARNING_RATE",
    "STEPS_PER_EPOCH",
    "NUM_EPOCHS",
    "WARMUP_STEPS",
    "GPU_BATCH_SIZE",
    "OPTMIZER_BETA1",
    "OPTMIZER_BETA2",
    "OPTMIZER_EPSILON",
    "ONLY_TEST",
    "FOLD",
    "SAMPLE_NORMALIZE",
    "DROP_INITIAL_DATE",
    "TRAIN",
    "PREDICT",
    "CFG",
    "DIM",
    "TAM_CONSIDER",
    # Distribution functions
    "gauss",
    "gauss_standard",
    "lognorm",
    "lognormal_standard",
    "student_t",
    # Utility functions
    "sample_normalize",
    "drop_initial_date",
    "truncate_days",
    # Data processing functions
    "read_data",
    "read_data_test",
    # Models
    "Encoder",
    "Model",
    # Loss functions
    "loss_function",
    "metrics",
    # Callbacks
    "CustomSchedule",
    "ReduceLROnThreshold",
    "MetricsCallback",
    # Evaluation functions
    "get_real_event",
    "build_preds_dict",
    "get_event",
    "get_scores",
    "get_events",
    "get_preds_df",
    # Training functions
    "get_inds",
    "read_indices",
    "read_val_indices",
    "reshape_features",
    "create_datasets",
    "prepare_validation_data",
    # Inference constants
    "REPLACE",
    "EXTEND",
    "LINEAR",
    "ATT",
    "CNN",
    "CONV1",
    "CONV_5",
    "CONV_6",
    "CONV1_LSTMDROPOUT",
    "CONV2",
    "GRU",
    "LINEAR_lstm_dropout",
    "DEFAULT_PATCH_SIZE",
    "W_BASE",
    "W_BASE_5",
    "W_BASE_8",
    "W_CONV",
    "VALIDATE",
    # Inference configs
    "CFG",
    "CFG_lstm_dropout",
    "CFG_Att",
    "CFG_CNN",
    "CFGConv1",
    "CFG_CONV1_LSTMDROPOUT",
    # Inference modelspec
    "ModelSpec",
    # Inference models
    "EncoderConv1",
    "ModelConv1",
    "EncoderConv6",
    "ModelConv6",
    "EncoderConv5",
    "ModelConv5",
    "EncoderConv1_LSTMDROPOUT",
    "ModelConv1_LSTMDROPOUT",
    "Encoder",
    "Model",
    "EncoderGRU",
    "ModelGRU",
    "Encoder_lstm_dropout",
    "Model_lstm_dropout",
    # Inference data processing
    "sample_normalize",
    "truncate_days",
    "drop_initial_date",
    "process_df",
    "read_data",
    "read_data_validate",
    # Inference prediction
    "get_event",
    "get_events",
    "get_preds_df",
    "get_df",
    "predict_model",
    # Inference losses
    "loss_function",
    "CustomSchedule",
]
