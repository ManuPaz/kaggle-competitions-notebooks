# AMP®-Parkinson's Disease Progression Prediction

## Competition Overview
- **Competition**: [AMP®-Parkinson's Disease Progression Prediction](https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction)
- **Position**: 11/1805
- **Medal**: 🥇 Gold Medal
- **Type**: Research · Code Competition
- **Teams**: 1805

## Competition Description
This competition focuses on predicting the progression of Parkinson's disease using clinical data, peptide abundance, and protein expression data. The goal is to predict UPDRS (Unified Parkinson's Disease Rating Scale) scores at different time points (0, 6, 12, and 24 months ahead) for four different UPDRS components (UPDRS 1-4).

**Metrics**: SMAPE (Symmetric Mean Absolute Percentage Error) for each UPDRS component and time horizon.

## 🎯 Key Strategy Insight - The Winning Approach

**The fundamental breakthrough of this competition was recognizing that peptide and protein features had minimal or no predictive power.** This led to a completely different approach that ultimately secured the Gold Medal position.

### Core Strategy: Time-Based Progression Modeling

Instead of relying on complex biological features, the winning solution focuses on **disease progression patterns over time**:

1. **Single Feature Model**: The final model uses only the `visit_month` as a feature
2. **Disease Evolution Prediction**: Predicts how the disease will progress at 4, 6, and 12 months from each visit
3. **Temporal Logic**: A patient at month 12 (4 months from visit) will have worse prognosis than a patient at month 0 (4 months from visit = month 4)

### Multi-Model Architecture: 24 Specialized Models

The solution creates **24 separate models** based on two key dimensions:

#### Dimension 1: UPDRS Components (4 models)
- **UPDRS_1**: Non-motor experiences of daily living
- **UPDRS_2**: Motor experiences of daily living  
- **UPDRS_3**: Motor examination
- **UPDRS_4**: Motor complications

#### Dimension 2: Prediction Horizons (3 timeframes)
- **4 months ahead**: Short-term progression
- **6 months ahead**: Medium-term progression
- **12 months ahead**: Long-term progression

#### Dimension 3: Patient Stratification (2 categories)
- **"Good" Patients**: Visit all months (better prognosis)
- **"Bad" Patients**: Not visit in some months (worse prognosis)

**Total: 4 UPDRS × 3 timeframes × 2 patient categories = 24 specialized models**

### Why This Approach Won

1. **Biological Reality**: Peptides and proteins showed no meaningful correlation with disease progression
2. **Temporal Patterns**: Disease progression follows predictable time-based patterns
3. **Patient Heterogeneity**: Different visit patterns indicate different disease trajectories
4. **Model Specialization**: Each model focuses on a specific prediction scenario

### Example Prediction Logic

For **UPDRS_1 at 4 months**:
- Patient visiting at month 0 → 4 months ahead = month 4 (better prognosis)
- Patient visiting at month 12 → 4 months ahead = month 16 (worse prognosis)

This temporal understanding was the key to climbing the leaderboard and achieving the Gold Medal position.

## Code Overview

### Directory Structure
- `parkinson.ipynb` - Main inference notebook containing the complete solution

### Key Features
- **Feature Engineering**: Advanced peptide and protein candidate selection based on coefficient of variation
- **Multi-model Approach**: Support for XGBoost, Linear Regression, and SVR models
- **Time Series Prediction**: Handles multiple prediction horizons (0, 6, 12, 24 months)
- **Patient Stratification**: Separate models for "good" and "bad" patients based on data availability

## Model Performance
- **Rank**: 11/1805
- **Medal**: 🥇 Gold Medal
- **Competition**: Highly competitive field with 1805 teams

## Technical Approach

### Data Processing
- **Peptide Selection**: Identifies top peptide candidates based on coefficient of variation in abundance across patients
- **Protein Selection**: Selects protein candidates based on NPX (Normalized Protein Expression) variability
- **Data Pivoting**: Transforms peptide and protein data into wide format for modeling
- **Missing Value Handling**: Implements forward-fill strategy for patient-specific missing values

### Model Architecture
**Not Deep Learning**: This solution uses traditional machine learning approaches:
- **XGBoost**: Primary model with optimized hyperparameters for regression
- **Support Vector Regression (SVR)**: Alternative model with linear and RBF kernels
- **Linear Regression**: Baseline model for comparison

**Algorithm Type**: 
- **Tabular Data**: Clinical, peptide, and protein features
- **Time Series**: Multi-horizon prediction (0, 6, 12, 24 months ahead)
- **Multi-target**: Four UPDRS components (UPDRS 1-4)

### Training Pipeline
- **Data Splitting**: Patient-level split to avoid data leakage
- **Feature Engineering**: 
  - Coefficient of variation calculation for peptide/protein selection
  - Time-based feature creation (visit_month)
  - Patient-specific data aggregation
- **Model Training**: Separate models for each UPDRS component and time horizon
- **Patient Stratification**: Different models for patients with complete vs. incomplete data

### Inference Pipeline
- **Streaming Prediction**: Handles Kaggle's streaming test environment
- **Model Selection**: Automatically selects appropriate model based on patient data quality
- **Post-processing**: 
  - Clips negative predictions to 0
  - Rounds predictions to nearest integer
  - Special handling for UPDRS 4 (always predicts 0)
- **Patient Classification**: Distinguishes between "good" and "bad" patients for appropriate model application

### Code Versions and Changes
The notebook contains a single, comprehensive solution with:
- **Model Selection**: Configurable model type (XGBoost, SVR, Linear)
- **Hyperparameter Tuning**: Optimized parameters for each model type
- **Adaptive Prediction**: Different strategies based on patient data availability

### Key Innovations
1. **Temporal Disease Progression Modeling**: Revolutionary approach using only visit_month as feature, recognizing that biological markers had no predictive value
2. **24-Specialized Model Architecture**: Separate models for each UPDRS component, time horizon, and patient category combination
3. **Patient Stratification by Visit Timing**: Classification of patients as "good" or "bad" based on optimal vs. suboptimal visit months
4. **Time-Based Prognosis Logic**: Understanding that disease progression follows predictable temporal patterns rather than biological markers
5. **Streaming Inference**: Efficient handling of Kaggle's real-time prediction environment

## Usage Instructions
1. Set `model_use` variable to desired model type ("xgboost", "svr", or "linear")
2. Adjust hyperparameters in respective parameter dictionaries
3. Run the complete notebook for training and inference
4. The solution automatically handles the streaming test environment

## Data Requirements
- Clinical data with patient visits and UPDRS scores
- Peptide abundance measurements
- Protein expression data (NPX values)
- Test data in streaming format

## Results
- **Gold Medal Performance**: 11th place out of 1805 teams
- **SMAPE Optimization**: Effective handling of multiple prediction horizons
- **Robust Prediction**: Consistent performance across different patient data scenarios

## Applications
This approach demonstrates effective feature engineering and model selection for:
- Medical time series prediction
- Multi-horizon forecasting
- Patient stratification in clinical data
- Streaming inference in production environments
