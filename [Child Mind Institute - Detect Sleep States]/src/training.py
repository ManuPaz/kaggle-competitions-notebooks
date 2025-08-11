"""
Training-related functions for the Child Mind Institute - Detect Sleep States project
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from .params import CFG, GPU_BATCH_SIZE


def get_inds(train_ids, val_ids, features):
    """
    Generate training and validation indices

    Args:
        train_ids: List of training IDs
        val_ids: List of validation IDs
        features: Features dictionary

    Returns:
        Tuple of (train_indices, val_indices, array)
    """
    arr = np.array(list(features.keys()))
    np.random.shuffle(arr)
    arr = [e for e in arr if e in train_ids]

    i = 0
    inds = []
    while len(inds) < 1_000_000:
        if len(features[arr[i]]) > 0:
            j = np.random.choice(range(len(features[arr[i]])))
            inds.append({"i": i, "j": j})

        i = i + 1
        i = i % len(arr)

    val_inds = []
    for i, e in enumerate(val_ids):
        for j in range(len(features[e])):
            val_inds.append({"i": i, "j": j})

    return inds, val_inds, arr


def read_indices(row, features, targets, arr):
    """
    Read training indices for TensorFlow dataset

    Args:
        row: Row from dataset
        features: Features dictionary
        targets: Targets dictionary
        arr: Array of IDs

    Returns:
        Tuple of (input, target)
    """

    def get_row(i, j):
        return features[arr[i]][j], targets[arr[i]][j]

    series_input, series_target = tf.py_function(get_row, [row["i"], row["j"]], [tf.float32, tf.float32])
    series_input.set_shape(shape=(CFG["block_size"] // CFG["patch_size"], CFG["patch_size"] * 2 + 2))
    series_target.set_shape(shape=(CFG["block_size"] // CFG["patch_size"], 2))

    return series_input, series_target


def read_val_indices(row, features, targets, val_ids):
    """
    Read validation indices for TensorFlow dataset

    Args:
        row: Row from dataset
        features: Features dictionary
        targets: Targets dictionary
        val_ids: Validation IDs

    Returns:
        Tuple of (input, target)
    """

    def get_row(i, j):
        return features[val_ids[i]][j], targets[val_ids[i]][j]

    series_input, series_target = tf.py_function(get_row, [row["i"], row["j"]], [tf.float32, tf.float32])
    series_input.set_shape(shape=(CFG["block_size"] // CFG["patch_size"], CFG["patch_size"] * 2 + 2))
    series_target.set_shape(shape=(CFG["block_size"] // CFG["patch_size"], 2))

    return series_input, series_target


def reshape_features(features, targets):
    """
    Reshape features and targets for training

    Args:
        features: Features dictionary
        targets: Targets dictionary

    Returns:
        Tuple of (reshaped_features, reshaped_targets)
    """
    for i in range(len(features)):
        for j in range(len(features[i])):
            features[i][j] = features[i][j].reshape(-1, features[i][j].shape[1] * CFG["patch_size"])
            feats = features[i][j][:, [0, 1, 4, 5, 8, 9]]
            sine_ = features[i][j][:, [2, 6, 10]].mean(axis=1).reshape(-1, 1)
            cosine_ = features[i][j][:, [3, 7, 11]].mean(axis=1).reshape(-1, 1)
            feats = np.concatenate([feats, sine_, cosine_], axis=1)
            features[i][j] = feats
            targets[i][j] = targets[i][j].reshape(-1, CFG["patch_size"], 2).mean(axis=1)

    return features, targets


def create_datasets(inds, val_inds, features, targets, val_ids):
    """
    Create TensorFlow datasets for training and validation

    Args:
        inds: Training indices
        val_inds: Validation indices
        features: Features dictionary
        targets: Targets dictionary
        val_ids: Validation IDs

    Returns:
        Tuple of (train_dataset, val_dataset, val_dataset_backup)
    """
    # Training dataset
    dataset = tf.data.Dataset.from_tensor_slices(dict(pd.DataFrame(inds)))
    dataset = dataset.map(lambda row: read_indices(row, features, targets, list(features.keys()))).batch(
        GPU_BATCH_SIZE, drop_remainder=False
    )

    # Validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices(dict(pd.DataFrame(val_inds)))
    val_dataset = val_dataset.map(lambda row: read_val_indices(row, features, targets, val_ids)).batch(
        GPU_BATCH_SIZE, drop_remainder=False
    )

    # Validation dataset for prediction (batch size 1)
    val_dataset_backup = tf.data.Dataset.from_tensor_slices(dict(pd.DataFrame(val_inds)))
    val_dataset__ = val_dataset_backup.map(lambda row: read_val_indices(row, features, targets, val_ids)).batch(
        1, drop_remainder=False
    )

    return dataset, val_dataset, val_dataset__


def prepare_validation_data(val_inds, GPU_BATCH_SIZE):
    """
    Prepare validation data with proper batch sizing

    Args:
        val_inds: Validation indices
        GPU_BATCH_SIZE: GPU batch size

    Returns:
        Tuple of (LEN_VAL, REMAINDER_VAL, padded_val_inds)
    """
    LEN_VAL = len(val_inds)
    REMAINDER_VAL = (GPU_BATCH_SIZE - len(val_inds) % GPU_BATCH_SIZE) % GPU_BATCH_SIZE

    # Pad validation indices to match batch size
    padded_val_inds = val_inds.copy()
    for i in range(REMAINDER_VAL):
        padded_val_inds.append(val_inds[-1])

    return LEN_VAL, REMAINDER_VAL, padded_val_inds
