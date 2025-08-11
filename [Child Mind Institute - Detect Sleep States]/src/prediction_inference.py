"""
Prediction functions for inference in the Child Mind Institute - Detect Sleep States project
"""

import numpy as np
import pandas as pd
import gc
from itertools import groupby
from tqdm import tqdm
from scipy.signal import find_peaks

from .constants_inference import VALIDATE
from .data_processing_inference import truncate_days, drop_initial_date


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


def get_events(
    pred,
    data_,
    idx,
    min_interval=30,
    patch_size=12,
    h=1.3429635e-07,
    d=30,
    TAM_CONSIDER=None,
):
    """
    Generate events from predictions
    
    Args:
        pred: Model predictions
        data_: Input data
        idx: Series ID
        min_interval: Minimum interval between events
        patch_size: Patch size
        h: Height threshold for peak detection
        d: Distance threshold for peak detection
        TAM_CONSIDER: TAM consideration factor
        
    Returns:
        DataFrame with events
    """
    test_ds = data_
    days = len(pred) / (17280 / 100)
    
    submission = pd.DataFrame(columns=["step", "event", "series_id", "score"])
    added_onset = np.array(find_peaks(pred[:, 0], height=h, distance=d)[0])
    added_wakeup = np.array(find_peaks(pred[:, 1], height=h, distance=d)[0])
    
    onset = (
        test_ds[["step"]]
        .iloc[np.clip(added_onset * TAM_CONSIDER, 0, len(test_ds) - 1)]
        .astype(np.int32)
    )
    onset["event"] = "onset"
    onset["series_id"] = idx
    onset["score"] = pred[added_onset, 0]
    
    wakeup = (
        test_ds[["step"]]
        .iloc[np.clip(added_wakeup * TAM_CONSIDER, 0, len(test_ds) - 1)]
        .astype(np.int32)
    )
    wakeup["event"] = "wakeup"
    wakeup["series_id"] = idx
    wakeup["score"] = pred[added_wakeup, 1]
    
    submission = pd.concat([submission, onset, wakeup], axis=0)
    return submission


def get_preds_df(val_ids, preds_dict, data, smoothing_length=480, TAM_CONSIDER=None):
    """
    Generate predictions DataFrame
    
    Args:
        val_ids: List of validation IDs
        preds_dict: Dictionary of predictions
        data: Input data
        smoothing_length: Smoothing length for predictions
        TAM_CONSIDER: TAM consideration factor
        
    Returns:
        DataFrame with predictions
    """
    submision = []
    
    for id_ in tqdm(val_ids):
        if len(preds_dict[id_]) > 0:
            preds = preds_dict[id_]
            
            df_ = data.loc[data.series_id == id_]
            df_ = truncate_days(df_, id_, None, None)  # events_check and dict_ids are None here
            data_ = df_
            
            if True and not len(data) == 450:  # DROP_INITIAL_DATE is True
                data_ = drop_initial_date(data_, None)  # model is None here
            
            submision.append(get_events(data_, id_, preds, TAM_CONSIDER=TAM_CONSIDER))
    
    submision = pd.concat(submision)
    submision["step"] = submision["step"].astype(np.float32)
    return submision


def get_df(final_preds, df_, model, TAM_CONSIDER):
    """
    Process final predictions with stride handling
    
    Args:
        final_preds: Final predictions
        df_: Input dataframe
        model: Model specification object
        TAM_CONSIDER: TAM consideration factor
        
    Returns:
        Processed predictions
    """
    if model.STRIDE != 0:
        offset = model.CFG["block_size"] - model.STRIDE
        final_preds_stride = []
        
        for preds__ in final_preds:
            if len(final_preds_stride) == 0:
                final_preds_stride.append(preds__)
            elif len(final_preds_stride) == 1:
                arr = np.concatenate(
                    [
                        final_preds_stride[len(final_preds_stride) - 1][
                            model.STRIDE :, 0
                        ].reshape(-1, 1),
                        preds__[: len(preds__) - model.STRIDE, 0].reshape(-1, 1),
                    ],
                    axis=1,
                )
                final_preds_stride[len(final_preds_stride) - 1][model.STRIDE :, 0] = (
                    np.nanmean(arr, axis=1)
                )
                
                arr = np.concatenate(
                    [
                        final_preds_stride[len(final_preds_stride) - 1][
                            model.STRIDE :, 1
                        ].reshape(-1, 1),
                        preds__[: len(preds__) - model.STRIDE, 1].reshape(-1, 1),
                    ],
                    axis=1,
                )
                final_preds_stride[len(final_preds_stride) - 1][model.STRIDE :, 1] = (
                    np.nanmean(arr, axis=1)
                )
                final_preds_stride.append(preds__[len(preds__) - model.STRIDE :])
                del arr
            else:
                arr = np.concatenate(
                    [
                        final_preds_stride[len(final_preds_stride) - 1][
                            model.STRIDE - offset :, 0
                        ].reshape(-1, 1),
                        preds__[: len(preds__) - model.STRIDE, 0].reshape(-1, 1),
                    ],
                    axis=1,
                )
                final_preds_stride[len(final_preds_stride) - 1][
                    model.STRIDE - offset :, 0
                ] = np.nanmean(arr, axis=1)
                
                arr = np.concatenate(
                    [
                        final_preds_stride[len(final_preds_stride) - 1][
                            model.STRIDE - offset :, 1
                        ].reshape(-1, 1),
                        preds__[: len(preds__) - model.STRIDE, 1].reshape(-1, 1),
                    ],
                    axis=1,
                )
                final_preds_stride[len(final_preds_stride) - 1][
                    model.STRIDE - offset :, 1
                ] = np.nanmean(arr, axis=1)
                final_preds_stride.append(preds__[len(preds__) - model.STRIDE :])
                del arr
            del preds__
        
        final_preds = final_preds_stride
        del final_preds_stride
        gc.collect()
    
    final_preds = np.concatenate(final_preds, axis=0)
    final_preds = final_preds[: TAM_CONSIDER * len(final_preds) // TAM_CONSIDER]
    
    if model.use_tam_consider:
        final_preds_r = np.mean(final_preds.reshape(-1, TAM_CONSIDER, 2), axis=1)
        del final_preds
        gc.collect()
        return final_preds_r * model.weight
    else:
        return final_preds * model.weight


def predict_model(models, TAM_CONSIDER=5, ids_test=None, VALIDATE=False, 
                 events=None, events_check=None, dict_ids=None, swapped_dict=None):
    """
    Main prediction function for multiple models
    
    Args:
        models: List of model specifications
        TAM_CONSIDER: TAM consideration factor
        ids_test: Test IDs
        VALIDATE: Validation flag
        events: Events dataframe
        events_check: Events check dataframe
        dict_ids: Dictionary mapping IDs
        swapped_dict: Swapped dictionary
        
    Returns:
        Submission dataframe
    """
    name_save = "submision.csv"
    
    # Load models
    for model_ in models:
        if model_.model_type == "linear":
            from .models_inference import Model
            model = Model(model_.CFG)
        elif model_.model_type == "att":
            from .models_inference import ModelATT
            model = ModelATT(model_.CFG)
        elif model_.model_type == "cnn":
            from .models_inference import ModelCNN
            model = ModelCNN(model_.CFG)
        elif model_.model_type == "conv1":
            from .models_inference import ModelConv1
            model = ModelConv1()
        elif model_.model_type == "LINEAR_lstm_dropout":
            from .models_inference import Model_lstm_dropout
            model = Model_lstm_dropout()
        elif model_.model_type == "CONV1_LSTMDROPOUT":
            from .models_inference import ModelConv1_LSTMDROPOUT
            model = ModelConv1_LSTMDROPOUT()
        elif model_.model_type == "CONV_6":
            from .models_inference import ModelConv6
            model = ModelConv6()
        elif model_.model_type == "CONV_5":
            from .models_inference import ModelConv5
            model = ModelConv5()
        elif model_.model_type == "GRU":
            from .models_inference import ModelGRU
            model = ModelGRU(model_.CFG)
        
        model.load_weights(model_.model_name)
        model_.model = model
    
    submisions = []
    submision = None
    df_total = None
    dfs_save = []
    
    # Process each test ID
    for id_ in tqdm(ids_test):
        model_ = models[0]
        
        if VALIDATE:
            from .data_processing_inference import read_data_validate
            features_test, descrp_test, steps_test, df_ = read_data_validate(
                id_, model_, events, events_check, dict_ids, swapped_dict
            )
        else:
            from .data_processing_inference import read_data
            features_test, descrp_test, steps_test, df_ = read_data(id_, model_)
        
        features_use = {}
        features_use[(model_.initial_hour, model_.patch_size)] = [
            features_test,
            descrp_test,
            steps_test,
            df_,
        ]
        
        if len(df_) > 0:
            matrix = np.stack(features_test[id_]).reshape(
                (-1, model_.CFG["block_size"] // model_.CFG["patch_size"], model_.dim)
            )
            
            preds_total = model_.model.predict(matrix, verbose=0)
            
            if model_.TAM_RESHAPE != 1:
                preds_total = np.tile(preds_total, model_.TAM_RESHAPE).reshape(
                    preds_total.shape[0], preds_total.shape[1] * model_.TAM_RESHAPE, 2
                )
            
            df_total = get_df(preds_total, df_, model_, TAM_CONSIDER=TAM_CONSIDER)
            
            # Process additional models
            for model_ in models[1:]:
                if (model_.initial_hour, model_.patch_size) in features_use.keys():
                    features_test, descrp_test, steps_test, df_ = features_use[
                        (model_.initial_hour, model_.patch_size)
                    ]
                else:
                    if VALIDATE:
                        features_test, descrp_test, steps_test, df_ = read_data_validate(
                            id_, model_, events, events_check, dict_ids, swapped_dict
                        )
                    else:
                        features_test, descrp_test, steps_test, df_ = read_data(id_, model_)
                    
                    features_use[(model_.initial_hour, model_.patch_size)] = [
                        features_test,
                        descrp_test,
                        steps_test,
                        df_,
                    ]
                
                matrix = np.stack(features_test[id_]).reshape(
                    (
                        -1,
                        model_.CFG["block_size"] // model_.CFG["patch_size"],
                        model_.dim,
                    )
                )
                
                preds = model_.model.predict(matrix, verbose=0)
                
                if model_.TAM_RESHAPE != 1:
                    preds = np.tile(preds, model_.TAM_RESHAPE).reshape(
                        preds.shape[0], preds.shape[1] * model_.TAM_RESHAPE, 2
                    )
                
                df__ = get_df(preds, df_, model_, TAM_CONSIDER=TAM_CONSIDER)
                
                if len(df__) < len(df_total):
                    df_total = df_total[: len(df__)]
                elif len(df__) > len(df_total):
                    df__ = df__[: len(df_total)]
                
                df_total += df__
            
            if VALIDATE:
                dfs_save.append((df_total.copy(), df_, id_))
            
            submision = get_events(
                df_total,
                df_,
                id_,
                patch_size=model_.patch_size,
                d=19,
                h=5e-6,
                TAM_CONSIDER=TAM_CONSIDER,
            )
            submisions.append(submision)
    
    # Save predictions
    import pickle as pkl
    df_save = dfs_save
    file_save = "preds_save.pkl"
    pkl.dump(df_save, open(file_save, "wb"))
    del df_save
    gc.collect()
    
    submision = pd.concat(submisions)
    submision["step"] = submision["step"].astype(np.float32)
    submision = submision.reset_index(drop=True).reset_index(names="row_id")
    
    return submision
