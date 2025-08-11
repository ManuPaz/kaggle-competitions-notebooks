# Kaggle - LLM Science Exam

## Competition Overview
- **Competition**: [Kaggle - LLM Science Exam](https://www.kaggle.com/competitions/kaggle-llm-science-exam)
- **Position**: 55/2664
- **Medal**: 🥈 Silver Medal
- **Type**: Featured · Code Competition
- **Teams**: 2664

## Competition Description
This competition challenges participants to use Large Language Models (LLMs) to answer difficult science questions. The goal is to develop models that can understand and respond to complex scientific queries across various domains including physics, chemistry, biology, and mathematics. The competition evaluates models on their ability to provide accurate, well-reasoned answers to challenging scientific questions.

**Metrics**: MAP@3 (Mean Average Precision at 3) for evaluating answer quality and relevance.

## Solution Overview

### Two Main Notebooks

#### 1. **Training Notebook** (`training/kaggle_llm_science_exam_eda_deberta.ipynb`)
- **Purpose**: Fine-tune DeBERTa model with layer freezing strategy
- **Key Features**:
  - **Model**: Microsoft DeBERTa-v3-large
  - **Strategy**: Freeze embedding layers and specific encoder layers for efficient training
  - **Data**: Training data augmented with scientific context
  - **Cross-Validation**: Stratified K-fold validation for robust model evaluation
  - **Optimization**: Custom training arguments and early stopping

#### 2. **Testing/Inference Notebook** (`inference/kaggle-llm-science-improved-retrivial.ipynb`)
- **Purpose**: Generate predictions using ensemble of 3 fine-tuned models with context retrieval
- **Key Features**:
  - **Context Retrieval**: TF-IDF based retrieval from Wikipedia dataset for enhanced context
  - **Ensemble Approach**: Combines predictions from 3 different fine-tuned models
  - **Model Weights**: 20% Model 1 + 60% Model 2 + 20% Model 3
  - **Context Integration**: Enhances questions with relevant scientific background
  - **Production Ready**: Generates competition submission files

### Directory Structure
- **`training/`**: Training notebook for DeBERTa fine-tuning
  - `kaggle_llm_science_exam_eda_deberta.ipynb` - DeBERTa fine-tuning with frozen layers
- **`inference/`**: Inference notebook with context retrieval and ensemble
  - `kaggle-llm-science-improved-retrivial.ipynb` - Context-aware inference with 3-model ensemble
- **`src/`**: Source code modules
  - `constants.py` - Configuration constants and paths
  - `params.py` - Training parameters and hyperparameters
  - `utils.py` - Utility functions and data collators
  - `metrics.py` - Competition metrics (Map@3)
  - `data_processing.py` - Data loading and preprocessing
  - `model.py` - Model setup and training configuration
  - `inference_utis.py` - Inference utilities and document retrieval

## Technical Approach

### Training Strategy (Notebook 1)
- **Model Architecture**: DeBERTa-v3-large (1.5B parameters)
- **Layer Freezing**: 
  - Freeze embedding layers to preserve pre-trained knowledge
  - Freeze specific encoder layers for efficient fine-tuning
  - Train only unfrozen layers on competition data
- **Data Augmentation**: Expand training data with scientific context
- **Cross-Validation**: Stratified K-fold for robust evaluation
- **Optimization**: Custom learning rates, batch sizes, and early stopping

### Inference Strategy (Notebook 2)
- **Context Retrieval System**:
  - **TF-IDF Vectorization**: N-gram (1,2) with sublinear TF scaling
  - **Knowledge Base**: Wikipedia dataset with parsed paragraphs
  - **Chunked Processing**: Memory-efficient processing of large datasets
  - **Top-K Retrieval**: Retrieve top 8 most relevant documents per question
- **Multi-Model Ensemble**:
  - **Model 1**: Fine-tuned on context-augmented data (20% weight)
  - **Model 2**: Fine-tuned on expanded dataset (60% weight) 
  - **Model 3**: Fine-tuned on validation-optimized data (20% weight)
- **Context Integration**: Combine retrieved context with questions for enhanced understanding

### Data Processing Pipeline
1. **Context Retrieval**: TF-IDF based search in Wikipedia dataset
2. **Text Preprocessing**: Unicode normalization, quotation removal, text cleaning
3. **Context Augmentation**: Add relevant scientific background to questions
4. **Tokenization**: Prepare inputs for multiple-choice model inference
5. **Ensemble Prediction**: Combine predictions from 3 models with weighted averaging

## Model Performance
- **Rank**: 55/2664 (🥈 Silver Medal)
- **Competition**: Highly competitive field with 2664 teams
- **Key Success Factors**:
  - Effective layer freezing strategy for efficient fine-tuning
  - Context retrieval system for enhanced question understanding
  - Multi-model ensemble for robust predictions
  - Data augmentation with scientific knowledge base

## Key Innovations
1. **Layer Freezing Strategy**: Preserve pre-trained knowledge while fine-tuning efficiently
2. **Context-Aware Training**: Incorporate scientific context for better understanding
3. **Advanced Retrieval System**: TF-IDF based document retrieval from Wikipedia dataset
4. **Multi-Model Ensemble**: Robust predictions through model combination
5. **Memory-Efficient Processing**: Chunked processing for large-scale datasets

## Usage Instructions

### For Training (Notebook 1)
1. **Setup**: Ensure access to DeBERTa-v3-large model
2. **Data**: Prepare training data with scientific context
3. **Configuration**: Adjust freezing parameters in `src/params.py`
4. **Training**: Run notebook to fine-tune model with frozen layers
5. **Evaluation**: Monitor cross-validation performance

### For Inference (Notebook 2)
1. **Models**: Load 3 fine-tuned models from different checkpoints
2. **Context**: Ensure Wikipedia dataset is available for retrieval
3. **Inference**: Run notebook to generate ensemble predictions
4. **Submission**: Generate competition submission file

## Data Requirements
- **Training Data**: Scientific question-answer pairs with context
- **Knowledge Base**: Wikipedia dataset for context retrieval
- **Pre-trained Models**: DeBERTa-v3-large for fine-tuning
- **Fine-tuned Models**: 3 different checkpoints for ensemble

## Results and Impact
- **Silver Medal Performance**: 55th place demonstrates effective approach
- **Efficient Training**: Layer freezing reduces computational requirements
- **Context Enhancement**: Wikipedia retrieval improves answer quality
- **Robust Predictions**: Ensemble approach provides stable performance

## Applications
This solution demonstrates effective approaches for:
- **Scientific Question Answering**: Context-aware LLM fine-tuning
- **Knowledge Retrieval**: TF-IDF based document search systems
- **Model Ensemble**: Multi-model prediction strategies
- **Efficient Fine-tuning**: Layer freezing for large language models
