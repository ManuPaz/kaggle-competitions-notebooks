"""
Constants and Configuration Variables

This module contains all the configuration constants and paths used throughout
the LLM Science Exam training pipeline. These constants define data paths,
model configuration, training parameters, and data processing settings.

Constants:
    DATA_PATH: Path to the data directory containing training files
    MODEL_NAME: Name of the pre-trained model to use
    MODEL_DIR: Directory where the fine-tuned model will be saved
    CHECKPOINT_DIR: Directory for saving training checkpoints
    OUTPUT_DIR: Output directory for model artifacts
    SEED: Random seed for reproducibility
    N_SPLITS: Number of cross-validation splits
    FOLDING: Whether to use cross-validation folding
    MAX_INPUT_LENGTH: Maximum input length for training data tokenization
    MAX_INPUT_LENGTH_VAL: Maximum input length for validation data tokenization
    USE_SAMPLE: Whether to use a sample of data for testing
    FREEZE_EMBEDDINGS: Whether to freeze model embeddings during training
    FREEZE_LAYERS: Number of model layers to freeze during training
    OPTIONS: String of answer options (A, B, C, D, E)
    INDICES: List of numerical indices corresponding to answer options
    STOP_WORDS: List of stop words for text processing
    REMOVE_COLS: List of columns to remove during data processing
"""

# Data paths
DATA_PATH = "C:/Users/manue/OneDrive/Documentos/kaggle-competitions-code/data/llm_science_exam"

# Model configuration
MODEL_NAME = "microsoft/deberta-v3-large"  # Pre-trained DeBERTa model for multiple choice tasks
MODEL_DIR = "finetuned_deberta"  # Directory to save the fine-tuned model
CHECKPOINT_DIR = "checkpoints"  # Directory for training checkpoints
OUTPUT_DIR = MODEL_DIR  # Output directory (same as model directory)

# Training configuration
SEED = 42  # Random seed for reproducibility
N_SPLITS = 5  # Number of cross-validation splits
FOLDING = False  # Whether to use cross-validation folding
MAX_INPUT_LENGTH = 450  # Maximum input length for training data tokenization
MAX_INPUT_LENGTH_VAL = 650  # Maximum input length for validation data tokenization

# Data processing
USE_SAMPLE = True  # Whether to use a sample of data for testing/development
FREEZE_EMBEDDINGS = True  # Whether to freeze model embeddings during training
FREEZE_LAYERS = 18  # Number of model layers to freeze during training

# Answer options mapping
OPTIONS = "ABCDE"  # String representation of answer choices
INDICES = list(range(5))  # Numerical indices [0, 1, 2, 3, 4] corresponding to A, B, C, D, E

# Stop words for text processing
STOP_WORDS = [
    "each",
    "other",
    "another",
    "any",
    "anybody",
    "anyone",
    "anything",
    "anywhere",
    "both",
    "either",
    "every",
    "everybody",
    "everyone",
    "everything",
    "everywhere",
    "few",
    "he",
    "her",
    "here",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "most",
    "my",
    "myself",
    "no",
    "nobody",
    "none",
    "nothing",
    "now",
    "nowhere",
    "of",
    "off",
    "on",
    "once",
    "one",
    "only",
    "or",
    "other",
    "others",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "some",
    "somebody",
    "someone",
    "something",
    "somewhere",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "therefore",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "within",
    "without",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "be",
]

# Columns to remove during data processing
REMOVE_COLS = [
    "prompt",  # Question prompt text
    "A",  # Answer option A
    "B",  # Answer option B
    "C",  # Answer option C
    "D",  # Answer option D
    "E",  # Answer option E
    "answer",  # Correct answer label
    "__index_level_0__",  # Pandas index column
    "context",  # Context information
    "source",  # Data source information
]
REMOVE_COLS_INFERENCE = [
    "prompt",
    "A",
    "B",
    "C",
    "D",
    "E",
    "answer",
    "__index_level_0__",
    "context",
    "source",
    "id",
]
