"""
ModelSpec class for inference in the Child Mind Institute - Detect Sleep States project
"""

from .constants_inference import (
    LINEAR, LINEAR_lstm_dropout, ATT, CNN, CONV1, CONV_5, CONV_6, 
    CONV1_LSTMDROPOUT, GRU
)
from .configs_inference import (
    CFG, CFG_lstm_dropout, CFG_Att, CFG_CNN, CFGConv1, CFG_CONV1_LSTMDROPOUT
)


class ModelSpec:
    """
    Class to specify model configuration for inference
    """
    
    def __init__(
        self,
        model_name,
        padding,
        use_temp,
        model_type,
        drop_initial_date,
        sample_normalize,
        initial_hour=12,
        patch_size=12,
        model_dim=320,
        lstm_layers=5,
        use_tam_consider=False,
        TAM_RESHAPE=1,
        weight=1,
        STRIDE=0,
    ):
        self.model_name = model_name
        self.padding = padding
        self.use_temp = use_temp
        self.model_type = model_type
        self.drop_initial_date = drop_initial_date
        self.sample_normalize = sample_normalize
        self.initial_hour = initial_hour
        self.use_tam_consider = use_tam_consider
        self.TAM_RESHAPE = TAM_RESHAPE
        
        # Set configuration based on model type
        if self.model_type == LINEAR:
            self.CFG = CFG.copy()
        elif self.model_type == LINEAR_lstm_dropout:
            self.CFG = CFG_lstm_dropout.copy()
        elif self.model_type == ATT:
            self.CFG = CFG_Att.copy()
        elif self.model_type == CNN:
            self.CFG = CFG_CNN.copy()
        elif self.model_type == CONV1:
            self.CFG = CFGConv1.copy()
        elif self.model_type == CONV_5:
            self.CFG = CFGConv1.copy()
        elif self.model_type == CONV_6:
            self.CFG = CFGConv1.copy()
        elif self.model_type == CONV1_LSTMDROPOUT:
            self.CFG = CFG_CONV1_LSTMDROPOUT.copy()
        elif self.model_type == GRU:
            self.CFG = CFG.copy()
        
        # Update configuration with custom parameters
        self.CFG["patch_size"] = patch_size
        self.CFG["model_dim"] = model_dim
        self.CFG["model_num_lstm_layers"] = lstm_layers
        
        if self.model_type != ATT:
            self.CFG["feature_mixing_dim"] = model_dim
        
        self.patch_size = patch_size
        
        # Calculate input dimension
        if self.use_temp:
            self.dim = (
                self.CFG["patch_size"] * 2 + 2
            )  # 2 numeric features * patch size + 2 more features (sine and cosine of day time)
        else:
            self.dim = self.CFG["patch_size"] * 2
        
        self.weight = weight
        self.STRIDE = STRIDE
        
        if self.STRIDE != 0:
            self.CFG["stride"] = self.STRIDE
