"""
Utility functions for data processing in the Child Mind Institute - Detect Sleep States project
"""

import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from pandas.core.arrays.timedeltas import timedelta


def sample_normalize(sample):
    """
    Normalize a sample by subtracting mean and dividing by standard deviation

    Args:
        sample: Input tensor/array to normalize

    Returns:
        Normalized sample as numpy array
    """
    mean = tf.math.reduce_mean(sample, axis=0)
    std = tf.math.reduce_std(sample, axis=0)
    sample = tf.math.divide_no_nan(sample - mean, std)
    return sample.numpy()


def drop_initial_date(df__):
    """
    Drop initial date and add padding if needed

    Args:
        df__: Input dataframe

    Returns:
        Processed dataframe with initial padding if needed
    """
    # Remove duplicate timestamps
    df__ = df__.loc[~(df__.timestamp.diff() == timedelta(seconds=0))]
    initial_time = df__.iloc[0].timestamp

    # Calculate initial padding
    initial_padding = (initial_time.hour * 60 + initial_time.minute - 60 * 12) * 12

    if initial_padding > 0:
        df_ini = pd.DataFrame(np.zeros((initial_padding, 9)), columns=df__.columns)
        df_ini["sleep"] = np.nan
        return pd.concat([df_ini, df__])
    else:
        return df__.iloc[-initial_padding:]


def truncate_days(df_, id_, events_check, dict_ids):
    """
    Truncate dataframe to match event dates

    Args:
        df_: Input dataframe
        id_: Series ID
        events_check: Events dataframe
        dict_ids: Dictionary mapping IDs

    Returns:
        Truncated dataframe
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        swapped_dict = {value: key for key, value in dict_ids.items()}
        events_ = events_check.loc[events_check.series_id == swapped_dict[id_]]
        events_["timestamp"] = pd.to_datetime(events_["timestamp"].str[:19])

        df_ = df_.loc[df_.timestamp.dt.date <= events_.timestamp.dt.date.max()]
        return df_
