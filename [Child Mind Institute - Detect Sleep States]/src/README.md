# Child Mind Institute - Detect Sleep States - Source Code

> **⚠️ IMPORTANT NOTE**: This code is **NOT directly executable** and serves as a **reference/example** only. The code requires significant adaptations, parameter tuning, and integration with your specific data pipeline and environment before it can be used for actual training or inference.

This directory contains the refactored source code for the Child Mind Institute - Detect Sleep States project, organized into modular Python files for better maintainability and reusability.

## File Structure

### Core Modules
- **`constants.py`** - Project constants and configuration values
- **`params.py`** - Model hyperparameters and training configuration
- **`distributions.py`** - Mathematical distribution functions (Gaussian, log-normal, Student's t)
- **`utils.py`** - General utility functions for data processing
- **`data_processing.py`** - Data reading and processing functions for training
- **`models.py`** - TensorFlow model architectures (Encoder, Model with LSTM layers)
- **`losses.py`** - Custom loss functions and metrics
- **`callbacks.py`** - Custom TensorFlow callbacks for training
- **`evaluation.py`** - Model evaluation and event generation functions
- **`training.py`** - Training pipeline and data preparation functions

### Inference Modules
- **`constants_inference.py`** - Constants specific to inference process
- **`configs_inference.py`** - Model configurations for different architectures
- **`modelspec_inference.py`** - ModelSpec class for flexible model configuration
- **`models_inference.py`** - All TensorFlow model classes for inference
- **`data_processing_inference.py`** - Data processing functions for inference
- **`prediction_inference.py`** - Prediction and event generation for inference
- **`losses_inference.py`** - Loss functions and schedulers for inference

## Key Concepts

### Target Smoothing Importance

**Target smoothing** is a critical technique in this sleep state detection project that significantly improves model performance and generalization. The importance lies in several key aspects:

1. **Biological Reality**: Sleep transitions are not instantaneous binary switches but gradual processes that occur over time. Raw binary labels (0/1) don't capture this biological reality.

2. **Noise Reduction**: Raw labels often contain annotation noise and inconsistencies. Smoothing helps reduce the impact of these labeling errors.

3. **Temporal Consistency**: Sleep states exhibit temporal dependencies - rapid changes between sleep/wake states are biologically implausible. Smoothing enforces this temporal consistency.

4. **Model Training Stability**: Smooth targets provide more stable gradients during training, leading to better convergence and preventing overfitting to noisy labels.

5. **Performance Improvement**: The project uses Gaussian smoothing with configurable sigma values (e.g., 720 time steps) to create smooth probability distributions that better represent the true underlying sleep state probabilities.

The smoothing approach transforms hard binary targets into continuous probability distributions, allowing the model to learn more nuanced representations of sleep-wake transitions and improving both training stability and inference accuracy.

### Training Approach

The training approach follows a sophisticated multi-stage strategy designed to handle the challenges of sleep state detection:

1. **Data Preparation Strategy**:
   - **Feature Engineering**: Creation of temporal features including sine/cosine transformations of time to capture circadian rhythms
   - **Normalization**: Standardization of sensor data (anglez, enmo) to ensure consistent scale across different subjects
   - **Sequence Processing**: Conversion of time series data into fixed-length sequences suitable for deep learning models

2. **Model Architecture Design**:
   - **LSTM-based Encoder**: Primary architecture using Long Short-Term Memory networks to capture temporal dependencies in sleep patterns
   - **Attention Mechanisms**: Incorporation of attention layers to focus on the most relevant time steps for sleep state classification
   - **Dropout Regularization**: Strategic use of dropout to prevent overfitting and improve generalization

3. **Training Strategy**:
   - **Learning Rate Scheduling**: Custom learning rate scheduler with warmup and decay phases for optimal convergence
   - **Dynamic LR Reduction**: Adaptive learning rate reduction based on validation loss thresholds to prevent overfitting
   - **Cross-validation**: K-fold cross-validation to ensure robust model evaluation and prevent data leakage
   - **Early Stopping**: Monitoring of validation metrics to stop training when performance plateaus

4. **Loss Function Design**:
   - **Custom Binary Cross-entropy**: Modified loss function with NaN masking to handle missing data gracefully
   - **Class Imbalance Handling**: Techniques to address the natural imbalance between sleep and wake states

5. **Validation Strategy**:
   - **Holdout Validation**: Careful separation of validation data to ensure unbiased performance estimation
   - **Multiple Metrics**: Evaluation using F1-score, accuracy, and confusion matrices for comprehensive performance assessment

### Ensemble Approach with Different Network Architectures

The inference phase employs a sophisticated ensemble approach that combines predictions from multiple neural network architectures to achieve superior performance:

1. **Architecture Diversity**:
   - **Convolutional Networks**: Multiple CNN variants (Conv1, Conv5, Conv6) with different kernel sizes and filter configurations
   - **Recurrent Networks**: LSTM and GRU architectures with various dropout configurations
   - **Hybrid Models**: Combinations of convolutional and recurrent layers for multi-scale feature extraction
   - **Attention-based Models**: Transformer-inspired architectures for capturing long-range temporal dependencies

2. **Ensemble Strategy**:
   - **Model Weighting**: Different models are assigned weights based on their individual performance and reliability
   - **Architecture-specific Weights**: Separate weight sets for different architectural families (base, convolutional, attention-based)
   - **Ensemble Averaging**: Final predictions are computed as weighted averages across all model outputs

3. **Benefits of Ensemble Approach**:
   - **Robustness**: Reduces the impact of individual model failures or biases
   - **Performance**: Combines the strengths of different architectures (CNNs for local patterns, RNNs for temporal dependencies)
   - **Generalization**: Better generalization to unseen data by leveraging diverse model perspectives
   - **Stability**: More consistent predictions across different subjects and conditions

4. **Implementation Details**:
   - **Model Loading**: Efficient loading of multiple pre-trained models with shared weights and configurations
   - **Parallel Processing**: Batch processing across multiple models for computational efficiency
   - **Prediction Aggregation**: Sophisticated aggregation algorithms that consider model confidence and historical performance

This ensemble approach represents a key innovation in the project, allowing it to leverage the complementary strengths of different neural network architectures while maintaining computational efficiency during inference.

## Usage Examples

### Training
```python
from src import read_data, create_datasets, Model, loss_function
from src import CustomSchedule, ReduceLROnThreshold, MetricsCallback

# Load and prepare data
train_data = read_data()
train_dataset, val_dataset = create_datasets(train_data)

# Create model
model = Model()
model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

# Train with custom callbacks
callbacks = [
    CustomSchedule(),
    ReduceLROnThreshold(),
    MetricsCallback()
]

model.fit(train_dataset, validation_data=val_dataset, callbacks=callbacks)
```

### Inference
```python
from src import predict_model, get_events, get_preds_df
from src import ModelSpec, CFG

# Configure model
model_spec = ModelSpec(
    model_type='CONV1',
    weights_path='path/to/weights',
    config=CFG
)

# Make predictions
predictions = predict_model(model_spec, test_data)
events = get_events(predictions)
results_df = get_preds_df(events)
```

## Dependencies

- TensorFlow 2.x
- NumPy
- Pandas
- Scipy
- Scikit-learn

## Notes

- All original notebooks remain unchanged
- Code has been extracted and organized for better maintainability
- Inference modules use `_inference` suffix to distinguish from training modules
- The ensemble approach combines multiple model architectures for improved performance
