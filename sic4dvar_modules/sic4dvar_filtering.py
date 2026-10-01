import copy
import logging
from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)
import numpy as np
import pandas as pd
import scipy
import sic4dvar_params as params
from lib.lib_dates import get_swot_dates, seconds_to_date_old
from lib.lib_swot_obs import filter_swot_obs_on_quality, get_flag_dict_from_config
from sic4dvar_functions.helpers.helpers_arrays import get_mask_nan_across_arrays, masked_array_to_nan_array

def filter_based_on_config(sic4dvar_dict):
    logging.info('Discard reaches and nodes depending on SWOT quality values')
    val_swot_q_flag = get_flag_dict_from_config(sic4dvar_dict['param_dict'])
    node_level_mask = filter_swot_obs_on_quality(sic4dvar_dict['flag_dict'], val_swot_q_flag)
    valid_z = np.sum(~np.isnan(sic4dvar_dict['input_data']['node_z']))
    valid_w = np.sum(~np.isnan(sic4dvar_dict['input_data']['node_w']))
    logging.info(f'Loaded SWOT data : {valid_z} valid wse and {valid_w} valid')
    sic4dvar_dict['output']['obs_filtering_ratio'] = np.sum(node_level_mask) * 100 // node_level_mask.size
    logging.info(f'Obs filtering ratio is {sic4dvar_dict['output']['obs_filtering_ratio']}%')
    sic4dvar_dict['input_data']['node_z'] = np.ma.masked_array(sic4dvar_dict['input_data']['node_z'], mask=node_level_mask)
    sic4dvar_dict['input_data']['node_w'] = np.ma.masked_array(sic4dvar_dict['input_data']['node_w'], mask=node_level_mask)
    valid_z = np.sum(~np.isnan(sic4dvar_dict['input_data']['node_z']))
    valid_w = np.sum(~np.isnan(sic4dvar_dict['input_data']['node_w']))
    logging.info(f'After obs node and reach level filtering : {valid_z} valid wse and {valid_w} valid')
    return sic4dvar_dict

def remove_unuseable_nodes(node_z0_array, node_w0_array, q_min_n_nodes_param: int=1, q_min_n_times_param: int=1, q_min_per_nodes_param: float=0.0, q_min_per_times_param: float=0.0) -> tuple:
    """
        Created on September 19th 2023 at 12:00
        by @Isadora Silva

        Last modified on September 19th 2023 12:00
        by @Isadora Silva

        @authors: Isadora Silva
        """
    '\n        raise IndexError in case no valid data is found\n\n        Returns\n        -------\n        tuple\n        (i) 1D np.array with the valid NODE indexes, (ii) 2D np.array with the valid TIME INSTANCES indexes\n\n        '
    Proceed = True
    if node_z0_array.shape != node_w0_array.shape:
        raise AssertionError('arrays should have the same shape')
    total_n_nodes, total_n_times = node_z0_array.shape
    node_z_array = masked_array_to_nan_array(copy.deepcopy(node_z0_array))
    node_w_array = masked_array_to_nan_array(copy.deepcopy(node_w0_array))
    node_w_array = np.where(node_w_array <= 0, np.nan, node_w_array)
    if np.all(np.isnan(node_z_array)):
        Proceed = False
        missing_indexes = [i for i in range(total_n_times)]
        logging.warning('All elevation data is invalid !')
        return (Proceed, [], [], missing_indexes)
    valid_cs_node_ids = (np.count_nonzero(np.isfinite(node_w_array), axis=1) > 2).nonzero()[0]
    if valid_cs_node_ids.size == 0:
        Proceed = False
        missing_indexes = [i for i in range(total_n_times)]
        logging.warning('All width data is invalid !')
        return (Proceed, [], [], missing_indexes)
    invalid_cs_node_ids = [i_ for i_ in range(total_n_nodes) if i_ not in valid_cs_node_ids]
    node_w_array[invalid_cs_node_ids, :] = np.nan
    nan_mask_array = get_mask_nan_across_arrays(node_z_array, node_w_array)
    if np.all(nan_mask_array):
        Proceed = False
        missing_indexes = [i for i in range(total_n_times)]
        logging.warning('No valid elevation data in any section at any time instant.')
        return (Proceed, [], [], missing_indexes)
    node_z_array[nan_mask_array] = np.nan
    node_z_array.shape = (total_n_nodes, total_n_times)
    df = pd.DataFrame(index=range(total_n_nodes), columns=range(total_n_times), data=node_z_array)
    original_shape = df.shape
    df = df.dropna(axis='columns', how='all')
    df = df.dropna(axis='rows', how='all')
    df = df.dropna(axis='columns', thresh=q_min_n_times_param)
    if total_n_times > 10:
        pd_thr = int(q_min_per_times_param / 100 * total_n_times)
        df = df.dropna(axis='columns', thresh=pd_thr)
    df = df.dropna(axis='rows', thresh=q_min_n_nodes_param)
    if total_n_nodes > 10:
        pd_thr = int(q_min_per_nodes_param / 100 * total_n_nodes)
        df = df.dropna(axis='rows', thresh=pd_thr)
    df = df.dropna(axis='columns', how='any')
    df = df.dropna(axis='rows', how='any')
    if df.empty:
        Proceed = False
        missing_indexes = [i for i in range(total_n_times)]
        logging.warning('Filtering out the missing data resulted in no valid elevation data in any nodes at any times')
        return (Proceed, [], [], missing_indexes)
    val_node_index_array, val_time_index_array = (df.index.to_numpy(), df.columns.to_numpy())
    missing_indexes = []
    for i in range(original_shape[1]):
        if i not in val_time_index_array:
            missing_indexes.append(i)
    return (Proceed, val_node_index_array, val_time_index_array, missing_indexes)