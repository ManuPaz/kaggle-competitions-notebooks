from __future__ import annotations

"""
Inference Utilities for LLM Science Exam Competition

This module provides comprehensive inference utilities for the LLM Science Exam competition,
including document retrieval, model inference, and ensemble prediction capabilities.

Key Features:
- TF-IDF based document retrieval from multiple knowledge sources
- Multi-model inference with ensemble prediction
- Context preparation and tokenization for multiple-choice questions
- Performance evaluation and submission generation

Functions:
- SplitList: Split lists into chunks of specified size
- get_relevant_documents_parsed: Retrieve documents using parsed paragraphs dataset
- get_relevant_documents: Retrieve documents using cohere dataset
- retrieval: Perform TF-IDF based document retrieval
- prepare_answering_input: Prepare input for answering with context
- softmax: Compute softmax probabilities

Dependencies:
- transformers: For model loading and inference
- torch: For tensor operations and model inference
- pandas: For data manipulation
- numpy: For numerical operations
- datasets: For dataset handling
- sklearn: For TF-IDF vectorization
"""

# Cell 0: Setup and imports
import os

os.chdir("../")

# Cell 1: Import constants and utilities from src
# Cell 2: Standard imports


import os
import unicodedata
import warnings

import numpy as np
from datasets import concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from src.constants import DATA_PATH, STOP_WORDS

warnings.filterwarnings("ignore")

# Cell 3: Check datasets version
import datasets

print(f"Datasets version: {datasets.__version__}")


# Cell 4: Utility functions for document retrieval
def SplitList(mylist, chunk_size):
    """
    Split a list into chunks of specified size.

    This utility function divides a list into smaller sublists, each containing
    up to `chunk_size` elements. Useful for processing large datasets in batches.

    Args:
        mylist (list): The input list to be split
        chunk_size (int): The maximum size of each chunk

    Returns:
        list: A list of lists, where each sublist contains up to `chunk_size` elements

    Example:
        >>> SplitList([1, 2, 3, 4, 5, 6], 2)
        [[1, 2], [3, 4], [5, 6]]
    """
    return [mylist[offs : offs + chunk_size] for offs in range(0, len(mylist), chunk_size)]


def get_relevant_documents_parsed(df_valid):
    """
    Retrieve relevant documents using parsed paragraphs dataset.

    This function loads and combines two datasets (parsed paragraphs and context paragraphs),
    then performs document retrieval for each validation question. It processes the data
    in chunks to manage memory efficiently.

    Args:
        df_valid (pd.DataFrame): Validation dataframe containing questions and options

    Returns:
        list: List of retrieved articles, where each article contains:
            - Relevance score
            - Article title
            - Article text content

    Note:
        - Uses a chunk size of 600 for processing
        - Combines parsed paragraphs and context paragraphs datasets
        - Applies text preprocessing to create searchable content
    """
    df_chunk_size = 600
    paraphs_parsed_dataset = load_from_disk(os.path.join(DATA_PATH, "all-paraphs-parsed-expanded"))
    my_context = load_from_disk(os.path.join(DATA_PATH, "context_paragraphs_204154.hf"))
    print(f"Parsed paragraphs: {len(paraphs_parsed_dataset)}, Context: {len(my_context)}")

    paraphs_parsed_dataset = concatenate_datasets([paraphs_parsed_dataset, my_context])
    print(f"Combined dataset size: {len(paraphs_parsed_dataset)}")

    modified_texts = paraphs_parsed_dataset.map(
        lambda example: {"temp_text": f"{example['title']} {example['section']} {example['text']}".replace("\n", " ")},
        num_proc=2,
    )["temp_text"]

    all_articles_indices = []
    all_articles_values = []
    for idx in tqdm(range(0, df_valid.shape[0], df_chunk_size)):
        df_valid_ = df_valid.iloc[idx : idx + df_chunk_size]
        articles_indices, merged_top_scores = retrieval(df_valid_, modified_texts)
        all_articles_indices.append(articles_indices)
        all_articles_values.append(merged_top_scores)

    article_indices_array = np.concatenate(all_articles_indices, axis=0)
    articles_values_array = np.concatenate(all_articles_values, axis=0).reshape(-1)

    top_per_query = article_indices_array.shape[1]
    articles_flatten = [
        (
            articles_values_array[index],
            paraphs_parsed_dataset[idx.item()]["title"],
            paraphs_parsed_dataset[idx.item()]["text"],
        )
        for index, idx in enumerate(article_indices_array.reshape(-1))
    ]
    retrieved_articles = SplitList(articles_flatten, top_per_query)
    return retrieved_articles


# Cell 5: Document retrieval functions
def get_relevant_documents(df_valid):
    """
    Retrieve relevant documents using cohere dataset.

    This function loads the cohere dataset and performs document retrieval for validation
    questions. It processes the data in chunks and applies text normalization for better
    matching.

    Args:
        df_valid (pd.DataFrame): Validation dataframe containing questions and options

    Returns:
        list: List of retrieved articles, where each article contains:
            - Relevance score
            - Article title
            - Normalized article text content

    Note:
        - Uses a chunk size of 800 for processing
        - Applies Unicode normalization (NFKD) to text
        - Removes quotation marks for better text matching
    """
    df_chunk_size = 800

    cohere_dataset_filtered = load_from_disk(os.path.join(DATA_PATH, "stem-wiki-cohere-no-emb"))
    modified_texts = cohere_dataset_filtered.map(
        lambda example: {
            "temp_text": unicodedata.normalize("NFKD", f"{example['title']} {example['text']}").replace('"', "")
        },
        num_proc=2,
    )["temp_text"]

    all_articles_indices = []
    all_articles_values = []
    for idx in tqdm(range(0, df_valid.shape[0], df_chunk_size)):
        df_valid_ = df_valid.iloc[idx : idx + df_chunk_size]
        articles_indices, merged_top_scores = retrieval(df_valid_, modified_texts)
        all_articles_indices.append(articles_indices)
        all_articles_values.append(merged_top_scores)

    article_indices_array = np.concatenate(all_articles_indices, axis=0)
    articles_values_array = np.concatenate(all_articles_values, axis=0).reshape(-1)

    top_per_query = article_indices_array.shape[1]
    articles_flatten = [
        (
            articles_values_array[index],
            cohere_dataset_filtered[idx.item()]["title"],
            unicodedata.normalize("NFKD", cohere_dataset_filtered[idx.item()]["text"]),
        )
        for index, idx in enumerate(article_indices_array.reshape(-1))
    ]
    retrieved_articles = SplitList(articles_flatten, top_per_query)
    return retrieved_articles


def retrieval(df_valid, modified_texts):
    """
    Perform TF-IDF based document retrieval.

    This function implements a TF-IDF based retrieval system that finds the most
    relevant documents for each question. It uses a two-stage approach: first fitting
    the vectorizer on validation data, then applying it to the knowledge base.

    Args:
        df_valid (pd.DataFrame): Validation dataframe containing questions and options
        modified_texts (list): List of preprocessed text documents from knowledge base

    Returns:
        tuple: (articles_indices, merged_top_scores)
            - articles_indices: Array of document indices for each query
            - merged_top_scores: Array of corresponding relevance scores

    Note:
        - Uses n-gram range (1, 2) for better phrase matching
        - Applies sublinear TF scaling for better score distribution
        - Processes documents in chunks of 100,000 for memory efficiency
        - Returns top 8 documents per query
    """
    corpus_df_valid = df_valid.apply(
        lambda row: f"{row['prompt']}\n{row['prompt']}\n{row['prompt']}\n{row['A']}\n{row['B']}\n{row['C']}\n{row['D']}\n{row['E']}",
        axis=1,
    ).values

    vectorizer1 = TfidfVectorizer(
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[\w/.-]+\b",
        stop_words=STOP_WORDS,
        sublinear_tf=True,
    )
    vectorizer1.fit(corpus_df_valid)
    vocab_df_valid = vectorizer1.get_feature_names_out()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[\w/.-]+\b",
        stop_words=STOP_WORDS,
        vocabulary=vocab_df_valid,
        sublinear_tf=True,
    )
    vectorizer.fit(modified_texts[:6_000_000])
    corpus_tf_idf = vectorizer.transform(corpus_df_valid)

    print(f"Length of vectorizer vocab: {len(vectorizer.get_feature_names_out())}")

    chunk_size = 100000
    top_per_chunk = 10
    top_per_query = 8

    all_chunk_top_indices = []
    all_chunk_top_values = []

    for idx in tqdm(range(0, len(modified_texts), chunk_size)):
        wiki_vectors = vectorizer.transform(modified_texts[idx : idx + chunk_size])
        temp_scores = (corpus_tf_idf * wiki_vectors.T).toarray()
        chunk_top_indices = temp_scores.argpartition(-top_per_chunk, axis=1)[:, -top_per_chunk:]
        chunk_top_values = temp_scores[np.arange(temp_scores.shape[0])[:, np.newaxis], chunk_top_indices]

        all_chunk_top_indices.append(chunk_top_indices + idx)
        all_chunk_top_values.append(chunk_top_values)

    top_indices_array = np.concatenate(all_chunk_top_indices, axis=1)
    top_values_array = np.concatenate(all_chunk_top_values, axis=1)

    merged_top_scores = np.sort(top_values_array, axis=1)[:, -top_per_query:]
    merged_top_indices = top_values_array.argsort(axis=1)[:, -top_per_query:]
    articles_indices = top_indices_array[np.arange(top_indices_array.shape[0])[:, np.newaxis], merged_top_indices]

    return articles_indices, merged_top_scores


# Cell 10: Inference with first model
def softmax(x):
    """
    Compute softmax probabilities.

    This function computes the softmax function over the input array, converting
    raw logits into probability distributions. It applies numerical stability
    by subtracting the maximum value before exponentiating.

    Args:
        x (np.ndarray): Input array of logits

    Returns:
        np.ndarray: Softmax probabilities that sum to 1 along the last axis

    Example:
        >>> softmax(np.array([1.0, 2.0, 3.0]))
        array([0.09003057, 0.24472847, 0.66524096])
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)
