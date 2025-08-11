# Predict Student Performance from Game Play

## Competition Overview
- **Competition**: [Predict Student Performance from Game Play](https://www.kaggle.com/competitions/predict-student-performance-from-game-play)
- **Position**: 35/2051
- **Medal**: 🥈 Silver Medal
- **Type**: Featured · Code Competition
- **Teams**: 2051

## 🎯 **Core Innovation: Advanced Feature Engineering with Polars**

This solution demonstrates the **critical importance of sophisticated feature engineering** in transforming raw, categorical gameplay events into powerful numerical features for machine learning. The key innovation lies in using **Polars** to efficiently process and transform categorical variables into hundreds of meaningful features that capture student behavior patterns.

### **The Feature Engineering Challenge**

The raw data consists of **categorical events** like:
- `event_name`: "click", "hover", "navigate"
- `fqid`: "tunic.historicalsociety.entry"
- `text`: "Hello there!" 
- `name`: "basic", "undefined", "close"

These categorical variables are **useless for ML models** in their raw form. The solution transforms them into **numerical features** through:

1. **Frequency Analysis**: Count of each event type per session
2. **Temporal Patterns**: Time-based aggregations (max, min, mean, sum)
3. **Spatial Analysis**: Coordinate-based features from room and screen positions
4. **Behavioral Sequences**: Pattern recognition in user interactions
5. **Cross-Feature Interactions**: Combinations of different event types

### **Why Polars is Critical**

- **Memory Efficiency**: Handles large datasets without memory issues
- **Speed**: Vectorized operations for rapid feature generation
- **Categorical Support**: Native handling of categorical variables
- **Groupby Operations**: Efficient aggregation per student session
- **Type Safety**: Prevents data type errors during transformations

## 🏗️ **Architecture & Code Structure**

### **Source Code (`src/`)**

#### **`constants.py`** - Feature Engineering Configuration
- **Data Type Definitions**: Optimized Polars data types for memory efficiency
- **Categorical Columns**: Lists of columns requiring transformation
- **Feature Lists**: Predefined elements for feature generation
- **Question Categorization**: Different modeling approaches per question type
- **Level Grouping**: Temporal boundaries for game progression analysis

#### **`feature_names.py`** - Generated Feature Registry
- **Question-Specific Features**: Final feature sets for questions 3, 12, 13
- **Feature Naming Convention**: `[element]_[metric]_[aggregation]`
- **Examples**: `"Hello there!_text_ET_max_", `"cutscene_click_event_name_counts"`

#### **`params.py`** - Optimized XGBoost Parameters
- **Question-Specific Tuning**: Different hyperparameters for each question
- **Optimized Thresholds**: Custom classification thresholds per question
- **Early Stopping**: Best iteration numbers to prevent overfitting

### **Inference Pipeline (`inference/`)**

#### **`xgb-using-previous-levels.ipynb`** - Main Inference Notebook
- **Feature Generation**: Transforms raw events into ML-ready features
- **Model Loading**: Loads pre-trained XGBoost models for each question
- **Prediction Pipeline**: Generates competition submission format

## 🚀 **Feature Engineering Pipeline**

### **Phase 1: Raw Data Processing**
```python
# Load raw gameplay events with optimized Polars data types
df = pl.read_csv(data_path, dtypes=DTYPES)
```

### **Phase 2: Categorical Transformation**
```python
# Transform categorical events into numerical features
for event in EVENT_NAME_FEATURE:
    features[f"{event}_counts"] = df.filter(pl.col("event_name") == event).count()
```

### **Phase 3: Temporal Aggregation**
```python
# Create time-based features per session
features[f"{text}_ET_max_"] = df.groupby("session_id").agg(
    pl.col("elapsed_time").max()
)
```

### **Phase 4: Spatial Analysis**
```python
# Extract coordinate-based patterns
features["room_coor_x_mean"] = df.groupby("session_id").agg(
    pl.col("room_coor_x").mean()
)
```

### **Phase 5: Behavioral Patterns**
```python
# Identify interaction sequences and patterns
features["navigation_pattern"] = analyze_navigation_sequence(df)
```

## 📊 **Model Architecture**

### **XGBoost Implementation**
- **Question-Specific Models**: Individual XGBoost model for each question
- **Custom Thresholds**: Optimized classification boundaries per question
- **Feature Selection**: Different feature sets for different question types
- **Cross-Validation**: 4-fold validation for robust performance estimation

### **Performance Optimization**
- **Tree Method**: "hist" for memory efficiency
- **Early Stopping**: Prevents overfitting with optimal iteration counts
- **Hyperparameter Tuning**: Question-specific optimization
- **Memory Management**: Efficient handling of large feature matrices

## ⚠️ **Important: Code Execution Status**

**The code is fully executable but will NOT generate predictions without trained models.**

### **What Works:**
- ✅ Feature engineering pipeline
- ✅ Data loading and preprocessing
- ✅ Feature generation from raw events
- ✅ Inference code structure
- ✅ Competition submission format

### **What's Missing:**
- ❌ Trained XGBoost models (not included in repository)
- ❌ Model weights and parameters
- ❌ Training data (competition data not publicly available)

### **To Generate Predictions:**
1. **Train Models**: Use the feature engineering pipeline to train XGBoost models
2. **Save Models**: Export trained models in XGBoost format
3. **Run Inference**: Execute the notebook with trained models loaded

## 🔬 **Technical Deep Dive**

### **Feature Engineering Techniques**

#### **1. Event Frequency Analysis**
- Count of each event type per session
- Frequency of specific text interactions
- Pattern recognition in user behavior

#### **2. Temporal Feature Extraction**
- Time between consecutive events
- Session duration analysis
- Event timing patterns

#### **3. Spatial Feature Generation**
- Room coordinate analysis
- Screen position patterns
- Navigation path analysis

#### **4. Behavioral Sequence Features**
- Click patterns and sequences
- Hover duration analysis
- Navigation flow patterns

### **Data Processing Pipeline**

```
Raw Events → Polars Processing → Feature Generation → Model Input
     ↓              ↓                ↓              ↓
Categorical    Type Conversion   Numerical      XGBoost
Variables      & Aggregation     Features       Prediction
```

## 📈 **Competition Results**

- **Final Rank**: 35/2051 (Top 1.7%)
- **Medal**: 🥈 Silver Medal
- **Key Success Factor**: Advanced feature engineering with Polars
- **Innovation**: Transformation of categorical events into meaningful numerical features

## 🎓 **Educational Applications**

This approach demonstrates effective machine learning for:
- **Educational Technology**: Game-based learning assessment
- **Student Analytics**: Behavioral pattern recognition
- **Personalized Learning**: Individual performance prediction
- **Learning Science**: Understanding student engagement patterns

## 🚀 **Getting Started**

### **Prerequisites**
```bash
pip install polars xgboost pandas numpy
```

### **Code Execution**
```bash
# Navigate to inference directory
cd inference/

# Run the main notebook
jupyter notebook xgb-using-previous-levels.ipynb
```

### **Expected Output**
- Feature engineering pipeline will execute successfully
- **No predictions will be generated** (models not included)
- Feature matrices will be created and saved
- Error messages about missing models (expected behavior)

## 🔍 **Key Insights**

1. **Feature Engineering is King**: Raw categorical data is transformed into 100+ meaningful features
2. **Polars Efficiency**: Critical for handling large-scale educational data
3. **Question-Specific Modeling**: Different approaches for different question types
4. **Behavioral Pattern Recognition**: Gameplay patterns predict academic performance
5. **Memory Optimization**: Efficient data types and processing prevent memory issues

## 📚 **Further Reading**

- [Polars Documentation](https://pola.rs/)
- [XGBoost Feature Engineering](https://xgboost.readthedocs.io/)
- [Educational Data Mining](https://educationaldatamining.org/)
- [Game-Based Learning Analytics](https://www.learntechlib.org/)

---

**Note**: This repository contains the complete feature engineering and inference pipeline that achieved a Silver Medal in the Kaggle competition. The code is production-ready but requires trained models to generate predictions.
