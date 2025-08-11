"""
Evaluation and scoring functions for the Child Mind Institute - Detect Sleep States project
"""

import warnings
from itertools import groupby

import numpy as np
import pandas as pd
from tqdm import tqdm

from .params import CFG, DROP_INITIAL_DATE
from .utils import drop_initial_date, truncate_days


def get_real_event(val_ids, DATA_PATH, dict_ids):
    """
    Get real events for validation IDs

    Args:
        val_ids: List of validation IDs
        DATA_PATH: Path to data directory
        dict_ids: Dictionary mapping IDs

    Returns:
        DataFrame with real events
    """
    events = pd.read_csv(f"{DATA_PATH}/train_events.csv")
    events["series_id"] = events["series_id"].transform(lambda x: dict_ids[x])
    events = events.loc[events.series_id.isin(val_ids)]
    return events


def build_preds_dict(val_ids, preds__, data, steps_, val_inds, n_data, dict_ids):
    """
    Build predictions dictionary for validation

    Args:
        val_ids: List of validation IDs
        preds__: Model predictions
        data: Input data
        steps_: Steps information
        val_inds: Validation indices
        n_data: Number of data points
        dict_ids: Dictionary mapping IDs

    Returns:
        Dictionary of predictions
    """
    preds_dict = {}

    for i, id_ in enumerate(val_ids):
        preds_dict[id_] = []
        df_ = data.loc[data.series_id == id_]
        df_ = truncate_days(df_, id_, None, dict_ids)  # events_check is None here

        if len(df_) > 0:
            if DROP_INITIAL_DATE:
                df_ = drop_initial_date(df_)

            step_ = 0
            for l, k in enumerate(val_inds[:n_data]):
                if k["i"] == i:
                    while step_ not in steps_[id_]:
                        if len(preds_dict[id_]) == 0:
                            preds_dict[id_].append(np.empty((int(CFG["block_size"] / CFG["patch_size"]), 2)) * np.nan)
                        elif len(preds_dict[id_]) == 1:
                            preds_dict[id_].append(np.empty(((CFG["stride"] // CFG["patch_size"]), 2)) * np.nan)
                        else:
                            preds_dict[id_].append(np.empty(((CFG["stride"] // CFG["patch_size"]), 2)) * np.nan)
                        step_ += CFG["stride"]

                    if len(preds__) > l:
                        if len(preds_dict[id_]) == 0:
                            preds_dict[id_].append(preds__[l])
                        elif len(preds_dict[id_]) == 1:
                            arr = np.concatenate(
                                [
                                    preds_dict[id_][len(preds_dict[id_]) - 1][
                                        CFG["stride"] // CFG["patch_size"] :, 0
                                    ].reshape(-1, 1),
                                    preds__[l][: len(preds__[l]) - CFG["stride"] // CFG["patch_size"], 0].reshape(
                                        -1, 1
                                    ),
                                ],
                                axis=1,
                            )

                            preds_dict[id_][len(preds_dict[id_]) - 1][CFG["stride"] // CFG["patch_size"] :, 0] = (
                                np.nanmax(arr, axis=1)
                            )

                            arr = np.concatenate(
                                [
                                    preds_dict[id_][len(preds_dict[id_]) - 1][
                                        CFG["stride"] // CFG["patch_size"] :, 1
                                    ].reshape(-1, 1),
                                    preds__[l][: len(preds__[l]) - CFG["stride"] // CFG["patch_size"], 1].reshape(
                                        -1, 1
                                    ),
                                ],
                                axis=1,
                            )

                            preds_dict[id_][len(preds_dict[id_]) - 1][CFG["stride"] // CFG["patch_size"] :, 1] = (
                                np.nanmax(arr, axis=1)
                            )

                            preds_dict[id_].append(preds__[l][len(preds__[l]) - CFG["stride"] // CFG["patch_size"] :])
                        else:
                            arr = np.concatenate(
                                [
                                    preds_dict[id_][len(preds_dict[id_]) - 1][
                                        CFG["stride"] // CFG["patch_size"] - 720 :, 0
                                    ].reshape(-1, 1),
                                    preds__[l][: len(preds__[l]) - CFG["stride"] // CFG["patch_size"], 0].reshape(
                                        -1, 1
                                    ),
                                ],
                                axis=1,
                            )

                            preds_dict[id_][len(preds_dict[id_]) - 1][CFG["stride"] // CFG["patch_size"] - 720 :, 0] = (
                                np.nanmax(arr, axis=1)
                            )

                            arr = np.concatenate(
                                [
                                    preds_dict[id_][len(preds_dict[id_]) - 1][
                                        CFG["stride"] // CFG["patch_size"] - 720 :, 1
                                    ].reshape(-1, 1),
                                    preds__[l][: len(preds__[l]) - CFG["stride"] // CFG["patch_size"], 1].reshape(
                                        -1, 1
                                    ),
                                ],
                                axis=1,
                            )

                            preds_dict[id_][len(preds_dict[id_]) - 1][CFG["stride"] // CFG["patch_size"] - 720 :, 1] = (
                                np.nanmax(arr, axis=1)
                            )

                            preds_dict[id_].append(preds__[l][len(preds__[l]) - CFG["stride"] // CFG["patch_size"] :])

                    step_ += CFG["stride"]

            if len(preds_dict[id_]) > 0:
                preds_dict[id_] = np.concatenate(preds_dict[id_], axis=0)

    return preds_dict


def get_event(df, col="pred"):
    """
    Convert predictions to event format

    Args:
        df: DataFrame with predictions
        col: Column name for predictions

    Returns:
        List of events
    """
    lstCV = zip(df.series_id, df[col])
    lstPOI = []

    for (c, v), g in groupby(lstCV, lambda cv: (cv[0], cv[1] != 0 and not pd.isnull(cv[1]))):
        llg = sum(1 for item in g)
        if v is False:
            lstPOI.extend([0] * llg)
        else:
            lstPOI.extend(["onset"] + (llg - 2) * [0] + ["wakeup"] if llg > 1 else [0])

    return lstPOI


def get_scores(val_ids, preds_dict, events, smoothing_lengths=[120], DATA_PATH=None, dict_ids=None):
    """
    Calculate scores for predictions

    Args:
        val_ids: List of validation IDs
        preds_dict: Dictionary of predictions
        events: Real events
        smoothing_lengths: List of smoothing lengths to try
        DATA_PATH: Path to data directory
        dict_ids: Dictionary mapping IDs

    Returns:
        Mean score across smoothing lengths
    """
    # Note: This function requires the 'score' function from evaluation.py
    # which is not available in the current context
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores_ = []

        for smoothing_length in smoothing_lengths:
            events_detected = {}
            solution = []

            for id_ in val_ids:
                if len(preds_dict[id_]) > 0:
                    events_ = events.loc[events.series_id == id_].dropna(subset="timestamp")
                    solution.append(events_)

            solution = pd.concat(solution).reset_index(drop=True)
            submision = get_preds_df(val_ids, preds_dict, None, smoothing_length, DATA_PATH, dict_ids)

            # Note: The 'score' function is not available here
            # This would need to be imported from the evaluation module
            print(f"Smoothing length {smoothing_length}: Score calculation requires evaluation module")
            scores_.append(0.0)  # Placeholder

        return np.mean(scores_)


def get_events(data_, idx, pred, min_interval=30):
    """
    Generate events from predictions

    Args:
        data_: Input data
        idx: Series ID
        pred: Predictions
        min_interval: Minimum interval between events

    Returns:
        DataFrame with events
    """
    test_ds = data_
    days = len(pred) / (17280 / 12)

    submission = pd.DataFrame(columns=["step", "event", "series_id", "score"])
    candidates_onset = np.argsort(-pred[:, 0])
    candidates_wakeup = np.argsort(-pred[:, 1])
    n_add = max(1, round(days))

    added_onset = []
    added_wakeup = []

    # Add onset events
    disponibles = list(candidates_onset.copy())
    while len(added_onset) < n_add and len(disponibles) > 0:
        actual = disponibles.pop(0)
        added_onset.append(actual)
        disponibles = [x for x in disponibles if abs(x - actual) >= min_interval]

    # Add wakeup events
    disponibles = list(candidates_wakeup.copy())
    while len(added_wakeup) < n_add and len(disponibles) > 0:
        actual = disponibles.pop(0)
        added_wakeup.append(actual)
        disponibles = [x for x in disponibles if abs(x - actual) >= min_interval]

    added_onset = np.array(added_onset)
    added_wakeup = np.array(added_wakeup)

    # Create onset events
    onset = test_ds[["step"]].iloc[np.clip(added_onset * CFG["patch_size"], 0, len(test_ds) - 1)].astype(np.int32)
    onset["event"] = "onset"
    onset["series_id"] = idx
    onset["score"] = pred[added_onset, 0]

    # Create wakeup events
    wakeup = test_ds[["step"]].iloc[np.clip(added_wakeup * CFG["patch_size"], 0, len(test_ds) - 1)].astype(np.int32)
    wakeup["event"] = "wakeup"
    wakeup["series_id"] = idx
    wakeup["score"] = pred[added_wakeup, 1]

    submission = pd.concat([submission, onset, wakeup], axis=0)
    return submission


def get_preds_df(val_ids, preds_dict, data, smoothing_length=480, DATA_PATH=None, dict_ids=None):
    """
    Generate predictions DataFrame

    Args:
        val_ids: List of validation IDs
        preds_dict: Dictionary of predictions
        data: Input data
        smoothing_length: Smoothing length for predictions
        DATA_PATH: Path to data directory
        dict_ids: Dictionary mapping IDs

    Returns:
        DataFrame with predictions
    """
    submision = []

    for id_ in tqdm(val_ids):
        if len(preds_dict[id_]) > 0:
            preds = preds_dict[id_]

            if data is not None:
                df_ = data.loc[data.series_id == id_]
                df_ = truncate_days(df_, id_, None, dict_ids)
                data_ = df_

                if DROP_INITIAL_DATE and not len(data) == 450:
                    data_ = drop_initial_date(data_)
            else:
                data_ = None

            submision.append(get_events(data_, id_, preds))

    submision = pd.concat(submision)
    submision["step"] = submision["step"].astype(np.float32)
    return submision
