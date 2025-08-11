"""
Data processing functions for inference in the Child Mind Institute - Detect Sleep States project
"""

import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import gc
from datetime import timedelta

from .constants import TARGET, NUMERIC_FEATURES
from .constants_inference import VALIDATE


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
        prev_length = len(df_)
        df_ = df_.loc[df_.timestamp.dt.date <= events_.timestamp.dt.date.max()]
        return df_


def drop_initial_date(df__, model):
    """
    Drop initial date and add padding if needed
    
    Args:
        df__: Input dataframe
        model: Model specification object
        
    Returns:
        Processed dataframe with initial padding if needed
    """
    if VALIDATE:
        df__ = df__.loc[~(df__.timestamp.diff() == timedelta(seconds=0))]
    
    initial_time = df__.iloc[0].timestamp
    
    if model.padding == "replace":
        initial_padding = (
            initial_time.hour * 60 + initial_time.minute - 60 * model.initial_hour
        ) * 12
    elif model.padding == "extend":
        initial_padding = (
            initial_time.hour * 60 + initial_time.minute
        ) * model.initial_hour + 12 * 60 * 12
    
    if initial_padding > 0:
        df_ini = pd.DataFrame(
            np.zeros((initial_padding, len(df__.columns))), columns=df__.columns
        )
        df_ini["sleep"] = np.nan
        return pd.concat([df_ini, df__])
    else:
        return df__.iloc[-initial_padding:]


def process_df(df_):
    """
    Process dataframe by adding timestamp, minute, sine, and cosine features
    
    Args:
        df_: Input dataframe
        
    Returns:
        Processed dataframe
    """
    df_["timestamp"] = pd.to_datetime(df_["timestamp"].str[:19])
    df_["minute"] = df_["timestamp"].dt.hour * 60 + df_["timestamp"].dt.minute
    df_["sine"] = np.sin(df_["minute"] * np.pi * 2 / 1440)
    df_["cosine"] = np.cos(df_["minute"] * np.pi * 2 / 1440)
    return df_


def read_data(id_, model):
    """
    Read and process data for inference
    
    Args:
        id_: Series ID
        model: Model specification object
        
    Returns:
        Tuple of (features, descrp, steps_, df_)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        features = {}
        targets = {}
        descrp = {}
        steps_ = {}
        preds_dic_len = {}
        
        features[id_] = []
        targets[id_] = []
        steps_[id_] = []
        
        df_ = pd.read_parquet(f"/kaggle/working/series/series_{id_}.parquet")
        df_ = process_df(df_)
        
        if len(df_) > 0:
            if model.drop_initial_date:
                df_ = drop_initial_date(df_, model)
            
            df_ = df_.reset_index(drop=True)
            preds_dic_len[id_] = len(df_)
            steps = range(0, len(df_), model.CFG["stride"])
            steps = [step for step in steps if step < len(df_)]
            descrp[id_] = {"steps": len(steps), "length_df": len(df_)}
            
            for step in steps:
                sample_ = df_.loc[step : step + model.CFG["block_size"] - 1, :]
                feats = sample_.loc[:, NUMERIC_FEATURES].values
                
                if model.sample_normalize:
                    feats = sample_normalize(feats)
                
                sine_ = sample_["sine"].values.reshape(-1, 1)
                cosine_ = sample_["cosine"].values.reshape(-1, 1)
                
                if len(feats) < model.CFG["block_size"]:
                    padding = model.CFG["block_size"] - len(feats)
                    padding_values = np.zeros((padding, feats.shape[1]))
                    padding_sine = np.zeros((padding, sine_.shape[1]))
                    padding_cosine = np.zeros((padding, cosine_.shape[1]))
                    
                    feats = np.vstack([feats, padding_values])
                    sine_ = np.vstack([sine_, padding_sine])
                    cosine_ = np.vstack([cosine_, padding_cosine])
                
                feats = feats.reshape(-1, model.CFG["patch_size"] * 2).astype(np.float32)
                sine_ = (
                    sine_.reshape(-1, model.CFG["patch_size"], 1)
                    .mean(axis=1)
                    .reshape(-1, 1)
                )
                cosine_ = (
                    cosine_.reshape(-1, model.CFG["patch_size"], 1)
                    .mean(axis=1)
                    .reshape(-1, 1)
                )
                
                if model.use_temp:
                    features[id_].append(
                        np.concatenate((feats, sine_, cosine_), axis=1)
                    )
                else:
                    features[id_].append(feats)
                
                steps_[id_].append(step)
                del sample_
            
            gc.collect()
    
    return features, descrp, steps_, df_


def read_data_validate(id_, model, events, events_check, dict_ids, swapped_dict):
    """
    Read and process data for validation
    
    Args:
        id_: Series ID
        model: Model specification object
        events: Events dataframe
        events_check: Events check dataframe
        dict_ids: Dictionary mapping IDs
        swapped_dict: Swapped dictionary
        
    Returns:
        Tuple of (features, descrp, steps_, df_)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        features = {}
        targets = {}
        descrp = {}
        steps_ = {}
        preds_dic_len = {}
        lengths = {}
        steps_save = {}
        
        features[id_] = []
        targets[id_] = []
        steps_[id_] = []
        
        df_ = pd.read_parquet(f"/kaggle/input/zzz-series/series/series_{id_}.parquet")
        df_["timestamp"] = pd.to_datetime(df_["timestamp"])
        df_ = truncate_days(df_, id_, events_check, dict_ids)
        
        events_ = events.loc[events.series_id == swapped_dict[id_]].dropna(subset="timestamp")
        
        if len(df_) > 0 and len(events_) > 0:
            if model.drop_initial_date:
                df_ = drop_initial_date(df_, model)
            
            df_ = df_.reset_index(drop=True)
            preds_dic_len[id_] = len(df_)
            steps = range(0, len(df_), model.CFG["stride"])
            steps = [step for step in steps if step < len(df_)]
            descrp[id_] = {"steps": len(steps), "length_df": len(df_)}
            
            for step in steps:
                sample_ = df_.loc[step : step + model.CFG["block_size"] - 1, :]
                feats = sample_.loc[:, NUMERIC_FEATURES].values
                
                if model.sample_normalize:
                    feats = sample_normalize(feats)
                
                sine_ = sample_["sine"].values.reshape(-1, 1)
                cosine_ = sample_["cosine"].values.reshape(-1, 1)
                target_ = sample_.loc[:, TARGET].values.reshape(-1, 1)
                
                if len(feats) < model.CFG["block_size"]:
                    padding = model.CFG["block_size"] - len(feats)
                    padding_values = np.zeros((padding, feats.shape[1]))
                    padding_sine = np.zeros((padding, sine_.shape[1]))
                    padding_cosine = np.zeros((padding, cosine_.shape[1]))
                    padding_target = np.empty((padding, cosine_.shape[1])) * np.nan
                    
                    feats = np.vstack([feats, padding_values])
                    sine_ = np.vstack([sine_, padding_sine])
                    cosine_ = np.vstack([cosine_, padding_cosine])
                    target_ = np.vstack([target_, padding_target])
                
                feats = feats.reshape(-1, model.CFG["patch_size"] * 2).astype(np.float32)
                sine_ = (
                    sine_.reshape(-1, model.CFG["patch_size"], 1)
                    .mean(axis=1)
                    .reshape(-1, 1)
                )
                cosine_ = (
                    cosine_.reshape(-1, model.CFG["patch_size"], 1)
                    .mean(axis=1)
                    .reshape(-1, 1)
                )
                targets_ = (
                    np.nanmean(target_.reshape(-1, model.CFG["patch_size"], 1), axis=1)
                    .round()
                    .astype(np.float16)
                )
                
                if True:  # Always add features for now
                    if model.use_temp:
                        features[id_].append(
                            np.concatenate((feats, sine_, cosine_), axis=1)
                        )
                    else:
                        features[id_].append(feats)
                    
                    steps_[id_].append(step)
                else:
                    df_ = df_.loc[
                        (df_.index < step)
                        | (df_.index >= step + model.CFG["block_size"])
                    ]
                
                del sample_
            
            gc.collect()
    
    return features, descrp, steps_, df_
