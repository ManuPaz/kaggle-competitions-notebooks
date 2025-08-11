import os
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.constants import DATA_PATH
from src.params import CFG, DROP_INITIAL_DATE, TAM_CONSIDER
from src.utils import drop_initial_date, truncate_days


def get_real_event(val_ids):
    events = pd.read_csv(os.path.join(DATA_PATH, "train_events.csv"))
    events["series_id"] = events["series_id"].transform(lambda x: dict_ids[x])
    events = events.loc[events.series_id.isin(val_ids)]
    return events


def build_preds_dict(val_ids, preds__, data, steps_, val_inds, n_data):
    preds_dict = {}
    for i, id_ in enumerate(val_ids):
        preds_dict[id_] = []
        df_ = data.loc[data.series_id == id_]
        df_ = truncate_days(df_, id_)
        # display(df_)
        if len(df_) > 0:
            if DROP_INITIAL_DATE:
                df_ = drop_initial_date(df_)

            l = 0

            step_ = 0
            for l, k in enumerate(val_inds[:n_data]):
                if k["i"] == i:
                    while step_ not in steps_[id_]:
                        if len(preds_dict[id_]) == 0:
                            # print("NAS in it 0")
                            preds_dict[id_].append(np.empty((int(CFG["block_size"] / CFG["patch_size"]), 2)) * np.nan)

                        elif len(preds_dict[id_]) == 1:
                            # print("NAS in it 1")
                            preds_dict[id_].append(np.empty(((CFG["stride"] // TAM_CONSIDER), 2)) * np.nan)
                        else:
                            preds_dict[id_].append(np.empty(((CFG["stride"] // TAM_CONSIDER), 2)) * np.nan)
                            # print("NAS in it != 0 or 1")
                        step_ += CFG["stride"]
                    if len(preds__) > l:
                        if len(preds_dict[id_]) == 0:
                            preds_dict[id_].append(preds__[l])
                        elif len(preds_dict[id_]) == 1:
                            # print(f"Lengh 1, {preds_dict[id_][len(preds_dict[id_])-1].shape}")
                            arr = np.concatenate(
                                [
                                    preds_dict[id_][len(preds_dict[id_]) - 1][
                                        CFG["stride"] // TAM_CONSIDER :, 0
                                    ].reshape(-1, 1),
                                    preds__[l][: len(preds__[l]) - CFG["stride"] // TAM_CONSIDER, 0].reshape(-1, 1),
                                ],
                                axis=1,
                            )
                            # print(len(preds__[l])-CFG["stride"]//TAM_CONSIDER)

                            preds_dict[id_][len(preds_dict[id_]) - 1][CFG["stride"] // TAM_CONSIDER :, 0] = np.nanmax(
                                arr, axis=1
                            )
                            arr = np.concatenate(
                                [
                                    preds_dict[id_][len(preds_dict[id_]) - 1][
                                        CFG["stride"] // TAM_CONSIDER :, 1
                                    ].reshape(-1, 1),
                                    preds__[l][: len(preds__[l]) - CFG["stride"] // TAM_CONSIDER, 1].reshape(-1, 1),
                                ],
                                axis=1,
                            )

                            preds_dict[id_][len(preds_dict[id_]) - 1][CFG["stride"] // TAM_CONSIDER :, 1] = np.nanmax(
                                arr, axis=1
                            )
                            # print(f"Lenght append {preds__[l][len(preds__[l])-CFG['stride']//TAM_CONSIDER:].shape}")
                            preds_dict[id_].append(preds__[l][len(preds__[l]) - CFG["stride"] // TAM_CONSIDER :])

                        else:
                            # print(f"Lengh > 1, {preds_dict[id_][len(preds_dict[id_])-1].shape}")
                            arr = np.concatenate(
                                [
                                    preds_dict[id_][len(preds_dict[id_]) - 1][
                                        CFG["stride"] // TAM_CONSIDER - 720 :, 0
                                    ].reshape(-1, 1),
                                    preds__[l][: len(preds__[l]) - CFG["stride"] // TAM_CONSIDER, 0].reshape(-1, 1),
                                ],
                                axis=1,
                            )

                            preds_dict[id_][len(preds_dict[id_]) - 1][CFG["stride"] // TAM_CONSIDER - 720 :, 0] = (
                                np.nanmax(arr, axis=1)
                            )
                            arr = np.concatenate(
                                [
                                    preds_dict[id_][len(preds_dict[id_]) - 1][
                                        CFG["stride"] // TAM_CONSIDER - 720 :, 1
                                    ].reshape(-1, 1),
                                    preds__[l][: len(preds__[l]) - CFG["stride"] // TAM_CONSIDER, 1].reshape(-1, 1),
                                ],
                                axis=1,
                            )

                            preds_dict[id_][len(preds_dict[id_]) - 1][CFG["stride"] // TAM_CONSIDER - 720 :, 1] = (
                                np.nanmax(arr, axis=1)
                            )
                            # print(f"Lenght append {preds__[l][len(preds__[l])-CFG['stride']//TAM_CONSIDER:].shape}")
                            preds_dict[id_].append(preds__[l][len(preds__[l]) - CFG["stride"] // TAM_CONSIDER :])

                    step_ += CFG["stride"]

            if len(preds_dict[id_]) > 0:
                preds_dict[id_] = np.concatenate(preds_dict[id_], axis=0)
    return preds_dict


def get_event(df, col="pred"):
    lstCV = zip(df.series_id, df[col])
    lstPOI = []
    for (c, v), g in groupby(lstCV, lambda cv: (cv[0], cv[1] != 0 and not pd.isnull(cv[1]))):
        llg = sum(1 for item in g)
        if v is False:
            lstPOI.extend([0] * llg)
        else:
            lstPOI.extend(["onset"] + (llg - 2) * [0] + ["wakeup"] if llg > 1 else [0])
    return lstPOI


def get_scores(val_ids, preds_dict, events, smoothing_lengths=[120]):
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
            submision = get_preds_df(val_ids, preds_dict, data, smoothing_length=smoothing_length)
            tolerances = {
                "onset": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
                "wakeup": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
            }
            score_ = score(
                solution,
                submision,
                tolerances,
                series_id_column_name="series_id",
                time_column_name="step",
                event_column_name="event",
                score_column_name="score",
            )
            print(smoothing_length, score_)
            scores_.append(score_)
        return np.mean(scores_)


def get_events(data_, idx, pred, min_interval=30):
    test_ds = data_

    days = len(pred) / (17280 / 12)

    submission = pd.DataFrame(columns=["step", "event", "series_id", "score"])
    candidates_onset = np.argsort(-pred[:, 0])
    candidates_wakeup = np.argsort(-pred[:, 1])
    n_add = max(1, round(days))

    added_onset = []
    added_wakeup = []
    disponibles = list(candidates_onset.copy())
    while len(added_onset) < n_add and len(disponibles) > 0:
        actual = disponibles.pop(0)
        added_onset.append(actual)
        disponibles = [x for x in disponibles if abs(x - actual) >= min_interval]

    disponibles = list(candidates_wakeup.copy())
    while len(added_wakeup) < n_add and len(disponibles) > 0:
        actual = disponibles.pop(0)
        added_wakeup.append(actual)
        disponibles = [x for x in disponibles if abs(x - actual) >= min_interval]
    added_onset = np.array(added_onset)
    added_wakeup = np.array(added_wakeup)
    onset = test_ds[["step"]].iloc[np.clip(added_onset * CFG["patch_size"], 0, len(test_ds) - 1)].astype(np.int32)
    onset["event"] = "onset"
    onset["series_id"] = idx
    onset["score"] = pred[added_onset, 0]
    wakeup = test_ds[["step"]].iloc[np.clip(added_wakeup * CFG["patch_size"], 0, len(test_ds) - 1)].astype(np.int32)
    wakeup["event"] = "wakeup"
    wakeup["series_id"] = idx
    wakeup["score"] = pred[added_wakeup, 1]
    submission = pd.concat([submission, onset, wakeup], axis=0)

    return submission


def get_preds_df(val_ids, preds_dict, data, smoothing_length=480):
    submision = []
    for id_ in tqdm(val_ids):
        if len(preds_dict[id_]) > 0:
            preds = preds_dict[id_]

            df_ = data.loc[data.series_id == id_]
            df_ = truncate_days(df_, id_)
            data_ = df_
            if DROP_INITIAL_DATE and not len(data) == 450:
                data_ = drop_initial_date(data_)

            submision.append(get_events(data_, id_, preds))
    submision = pd.concat(submision)
    submision["step"] = submision["step"].astype(np.float32)
    return submision
