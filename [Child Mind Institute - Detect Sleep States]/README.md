# Child Mind Institute - Detect Sleep States

## Competition Overview
- **Competition**: [Child Mind Institute - Detect Sleep States](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states)
- **Position**: 179/1877
- **Medal**: 🥉 Bronze Medal
- **Type**: Featured · Code Competition
- **Teams**: 1877

## Competition Description
This competition focuses on detecting sleep onset and wake states from wrist-worn accelerometer data. The goal is to develop machine learning models that can accurately identify when subjects are sleeping or awake based on movement patterns captured by wearable devices. This has important applications in sleep research, health monitoring, and behavioral studies.

**Metrics**: Event Detection Average Precision (AP) for sleep state detection accuracy.

## Code Overview

### Directory Structure
- **`training/`**: Contains training notebooks and model development
  - `create_target.ipynb` - Training notebook with neural network baseline
  - `train.ipynb` - Main training notebook for model development
  - `model_versions/` - Directory for different model iterations

### Key Features
- **Neural Network Baseline**: Multi-attention and bidirectional LSTM architecture
- **Time Series Analysis**: Processing of accelerometer data sequences
- **Event Detection**: Identification of sleep onset and wake events
- **Model Training**: Comprehensive training pipeline for sleep detection

## Model Performance
- **Rank**: 179/1877
- **Medal**: 🥉 Bronze Medal
- **Competition**: Competitive field with 1877 teams

## Technical Approach

### Data Processing
- **Accelerometer Data**: Processing of 3-axis movement data from wrist-worn devices
- **Time Series Segmentation**: Breaking continuous data into manageable chunks
- **Feature Extraction**: Statistical and temporal features from movement patterns
- **Target Creation**: Binary classification of sleep vs. wake states

### Model Architecture
**Deep Learning**: This solution uses advanced neural network architectures:
- **Multi-Attention Networks**: Captures temporal dependencies in movement patterns
- **Bidirectional LSTM**: Processes accelerometer sequences in both directions
- **2D Matrix Reshaping**: Converts time series data into 2D matrices for CNN processing

**Algorithm Type**: 
- **Time Series Analysis**: Sequential accelerometer data processing
- **Event Detection**: Binary classification of sleep states
- **Signal Processing**: Movement pattern analysis from wearable devices

**Neural Network Type**:
- **LSTM (Long Short-Term Memory)**: Handles temporal dependencies in movement data
- **Attention Mechanisms**: Focuses on relevant time periods for sleep detection
- **Convolutional Layers**: Processes 2D representations of time series data
- **Bidirectional Processing**: Captures both forward and backward temporal relationships

### Training Pipeline
- **Data Preparation**: 
  - Loading accelerometer data from parquet files
  - Creating target distributions for sleep states
  - Handling missing data and edge cases
- **Model Training**: 
  - Neural network with multi-attention and bidirectional LSTM
  - Loss function with masking for unknown states
  - Model checkpointing and validation
- **Feature Engineering**: 
  - Statistical features from accelerometer data
  - Temporal features capturing movement patterns
  - Data augmentation for robust training

### Code Versions and Changes
The solution includes training notebooks for model development:
- **Target Creation**: `create_target.ipynb`
  - Target variable preparation for training
  - Data preprocessing and feature engineering
  - Initial model setup and configuration
- **Training**: `train.ipynb`
  - Main training notebook for neural network development
  - Multi-attention and bidirectional LSTM architecture
  - Model training, validation, and checkpointing

### Key Innovations
1. **Multi-Attention Architecture**: Novel approach to capture temporal dependencies
2. **Bidirectional LSTM**: Processes accelerometer data in both temporal directions
3. **2D Matrix Representation**: Converts time series to 2D for CNN processing
4. **Masked Loss Function**: Handles unknown sleep states during training
5. **Efficient Training Pipeline**: Streamlined approach for model development

## Usage Instructions
1. **Target Preparation**: Run `create_target.ipynb` to prepare target variables and data
2. **Training**: Use `train.ipynb` to train sleep detection models
3. **Model Loading**: Load trained models from the model_versions directory
4. **Data Requirements**: Ensure accelerometer data is in the correct format

## Data Requirements
- Wrist-worn accelerometer data (3-axis movement)
- Sleep state annotations for training
- Pre-trained neural network models

## Results
- **Bronze Medal Performance**: 179th place out of 1877 teams
- **Advanced Time Series**: State-of-the-art neural network for sleep detection
- **Event Detection**: Effective identification of sleep onset and wake events
- **Efficient Training**: Streamlined pipeline for reliable model development

## Applications
This approach demonstrates effective deep learning for:
- Sleep research and monitoring
- Wearable device analytics
- Behavioral pattern recognition
- Health monitoring systems
- Time series event detection
