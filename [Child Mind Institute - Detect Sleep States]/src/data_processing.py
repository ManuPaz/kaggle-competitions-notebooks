"""
Data processing functions for the Child Mind Institute - Detect Sleep States project
"""

import gc
import warnings

import numpy as np
from tqdm import tqdm

from .constants import NUMERIC_FEATURES, TARGET
from .distributions import lognormal_standard
from .params import CFG, DROP_INITIAL_DATE, SAMPLE_NORMALIZE
from .utils import drop_initial_date, sample_normalize, truncate_days


def read_data(data, ids_, targets_events, events_check, dict_ids):
    """
    Read and process training data

    Args:
        data: Input dataframe
        ids_: List of series IDs
        targets_events: Dictionary of target events
        events_check: Events dataframe
        dict_ids: Dictionary mapping IDs

    Returns:
        Tuple of (features, targets, descrp, steps_)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        features = {}
        targets = {}
        descrp = {}
        steps_ = {}

        for id_ in tqdm(ids_):
            features[id_] = []
            targets[id_] = []
            steps_[id_] = []
            df_ = data.loc[data.series_id == id_]
            df_ = truncate_days(df_, id_, events_check, dict_ids)

            if len(df_) > 0:
                if DROP_INITIAL_DATE:
                    df_ = drop_initial_date(df_)

                df__ = df_.reset_index()
                steps = range(0, len(df_), CFG["stride"])
                steps = [step for step in steps if step < len(df_)]
                descrp[id_] = {"steps": len(steps), "length_df": len(df_)}

                for step in steps:
                    sample_ = df_.iloc[step : step + CFG["block_size"], :]
                    feats = sample_.loc[:, NUMERIC_FEATURES].values

                    if SAMPLE_NORMALIZE:
                        feats = sample_normalize(feats)

                    sine_ = sample_["sine"].values.reshape(-1, 1)
                    cosine_ = sample_["cosine"].values.reshape(-1, 1)
                    target_ = np.zeros((len(sample_.loc[:, TARGET].values), 2))
                    target_points = targets_events[id_]

                    for s_, e_ in target_points:
                        s = df__.loc[df__.step == s_].index[0]
                        e = df__.loc[df__.step == e_].index[0]

                        if s >= step and s < step + CFG["block_size"]:
                            s = s - step
                            st1, st2 = max(0, s - 360), min(len(target_), s + 360 + 1)
                            print(f"Added onset {s} in step {step}, affecting interval {st1}:{st2}")
                            target_[st1:st2, 0] = lognormal_standard()[
                                st1 - (s - 360) : 720 + 1 - ((s + 360 + 1) - st2)
                            ]

                        if e >= step and e < step + CFG["block_size"]:
                            e = e - step
                            ed1, ed2 = max(0, e - 360), min(len(target_), e + 360 + 1)
                            print(f"Added wakeup {e} in step {step}, affecting interval {ed1}:{ed2}")
                            target_[ed1:ed2, 1] = lognormal_standard()[
                                ed1 - (e - 360) : 720 + 1 - ((e + 360 + 1) - ed2)
                            ]

                        gc.collect()
                        if e > step + CFG["block_size"]:
                            print(f"Finished {id_}, step {step + CFG['block_size']}")
                            break

                    target_ = target_.reshape(-1, 2)

                    if len(feats) < CFG["block_size"]:
                        padding = CFG["block_size"] - len(feats)
                        padding_values = np.zeros((padding, feats.shape[1]))
                        padding_sine = np.zeros((padding, sine_.shape[1]))
                        padding_cosine = np.zeros((padding, cosine_.shape[1]))
                        padding_target = np.empty((padding, 2)) * np.nan
                        feats = np.vstack([feats, padding_values])
                        sine_ = np.vstack([sine_, padding_sine])
                        cosine_ = np.vstack([cosine_, padding_cosine])
                        target_ = np.vstack([target_, padding_target])

                    feats = feats.reshape(-1, CFG["patch_size"] * 2).astype(np.float32)
                    sine_ = sine_.reshape(-1, CFG["patch_size"], 1).mean(axis=1).reshape(-1, 1)
                    cosine_ = cosine_.reshape(-1, CFG["patch_size"], 1).mean(axis=1).reshape(-1, 1)
                    targets_ = np.nanmean(target_.reshape(-1, CFG["patch_size"], 2), axis=1).astype(np.float16)

                    if max(targets_.max(axis=0)) != 0:
                        print(f"MAX IS {max(targets_.max(axis=0))}")
                        features[id_].append(np.concatenate((feats, sine_, cosine_), axis=1))
                        targets[id_].append(targets_)
                        steps_[id_].append(step)
                    else:
                        print("MAX IS 0")

                    del sample_

                del df_
                gc.collect()

        return features, targets, descrp, steps_


def read_data_test(data, ids_, events_check, dict_ids):
    """
    Read and process test data

    Args:
        data: Input dataframe
        ids_: List of series IDs
        events_check: Events dataframe
        dict_ids: Dictionary mapping IDs

    Returns:
        Tuple of (features, descrp, steps_)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        features = {}
        descrp = {}
        steps_ = {}

        for id_ in tqdm(ids_):
            features[id_] = []
            steps_[id_] = []
            df_ = data.loc[data.series_id == id_]

            if DROP_INITIAL_DATE:
                df_ = drop_initial_date(df_)

            steps = range(0, len(df_), CFG["stride"])
            steps = [step for step in steps if step < len(df_)]
            descrp[id_] = {"steps": len(steps), "length_df": len(df_)}

            for step in steps:
                sample_ = df_.iloc[step : step + CFG["block_size"], :]
                feats = sample_.loc[:, NUMERIC_FEATURES].values

                if SAMPLE_NORMALIZE:
                    feats = sample_normalize(feats)

                sine_ = sample_["sine"].values.reshape(-1, 1)
                cosine_ = sample_["cosine"].values.reshape(-1, 1)

                if len(feats) < CFG["block_size"]:
                    padding = CFG["block_size"] - len(feats)
                    padding_values = np.zeros((padding, feats.shape[1]))
                    padding_sine = np.zeros((padding, sine_.shape[1]))
                    padding_cosine = np.zeros((padding, cosine_.shape[1]))
                    feats = np.vstack([feats, padding_values])
                    sine_ = np.vstack([sine_, padding_sine])
                    cosine_ = np.vstack([cosine_, padding_cosine])

                feats = feats.reshape(-1, CFG["patch_size"] * 2).astype(np.float32)
                sine_ = sine_.reshape(-1, CFG["patch_size"], 1).mean(axis=1).reshape(-1, 1)
                cosine_ = cosine_.reshape(-1, CFG["patch_size"], 1).mean(axis=1).reshape(-1, 1)

                features[id_].append(np.concatenate((feats, sine_, cosine_), axis=1))
                steps_[id_].append(step)

                del sample_

            del df_
            gc.collect()

        return features, descrp, steps_
