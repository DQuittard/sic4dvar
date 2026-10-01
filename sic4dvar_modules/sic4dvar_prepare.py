import datetime
import logging
from copy import deepcopy
from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)
import numpy as np
import pandas as pd
import sic4dvar_params as params
from lib.lib_dates import seconds_to_date_old
from lib.lib_verif import check_na
from sic4dvar_functions import sic4dvar_calculations as calc
import datetime

def force_create_reach_times_from_nodes(swot_dict):
    test_t_array = np.ones(len(swot_dict['input_data']['node_t'][0]))
    for t in range(0, len(swot_dict['input_data']['node_t'][0])):
        for n in range(0, len(swot_dict['input_data']['node_t'])):
            if check_na(swot_dict['input_data']['node_t'][n, t]):
                swot_dict['input_data']['node_t'][n, t] = np.nan
        test_t_array[t] = np.nanmean(swot_dict['input_data']['node_t'][:, t])
    return test_t_array

def update_sic4dvar_dict(sic4dvar_dict, keep_indices):
    reach_keys = ['reach_dA', 'reach_s', 'reach_w', 'reach_z', 'reach_t']
    node_keys = ['node_w', 'node_z', 'node_dA', 'node_s', 'node_t']
    for key in reach_keys:
        if key in sic4dvar_dict['input_data'] and len(np.shape(sic4dvar_dict['input_data'][key])) > 0:
            sic4dvar_dict['input_data'][key] = sic4dvar_dict['input_data'][key][keep_indices]
    for key in node_keys:
        if key in sic4dvar_dict['input_data'] and len(np.shape(sic4dvar_dict['input_data'][key])) == 2:
            sic4dvar_dict['input_data'][key] = sic4dvar_dict['input_data'][key][:, keep_indices]

def remove_duplicate_times(sic4dvar_dict):
    reach_t = sic4dvar_dict['input_data']['reach_t']
    fill_value = -999999999999.0
    s_reach_t = pd.Series(reach_t)
    duplicate_mask = s_reach_t.duplicated(keep='first').to_numpy()
    nan_mask = s_reach_t.isna().to_numpy()
    valid_mask = ~nan_mask
    keep_mask = ~duplicate_mask & valid_mask
    drop_mask = duplicate_mask & valid_mask
    keep_indices = np.where(keep_mask)[0]
    drop_indices = np.where(drop_mask)[0]
    nan_indices = np.where(nan_mask)[0]
    drop_to_keep_indices = np.array([], dtype=int)
    if drop_indices.size > 0:
        first_keep_by_value = s_reach_t[keep_mask].groupby(s_reach_t[keep_mask]).apply(lambda x: x.index[0])
        drop_to_keep_indices = s_reach_t.iloc[drop_indices].map(first_keep_by_value).to_numpy(dtype=int)
    for d, k in zip(drop_indices, drop_to_keep_indices):
        reach_t[d] = np.nan
    sic4dvar_dict['input_data']['reach_t'] = deepcopy(reach_t)
    return sic4dvar_dict

def prepare_params(input_data, flag_dict, param_dict, params):
    sic4dvar_dict = {'input_data': input_data, 'flag_dict': flag_dict, 'param_dict': param_dict, 'filtered_data': {}, 'output': {}, 'algo5_results': {}, 'bb': 9999.0, 'reliability': 'valid', 'stopped_stage': None}
    sic4dvar_dict['input_data']['series_cv'] = None
    if sic4dvar_dict['param_dict']['run_type'] == 'seq' and params.force_create_reach_t:
        sic4dvar_dict['input_data']['reach_t'] = force_create_reach_times_from_nodes(sic4dvar_dict)
        sic4dvar_dict = remove_duplicate_times(sic4dvar_dict)
    if sic4dvar_dict['param_dict']['run_type'] == 'set':
        test_t_array = np.ones(len(sic4dvar_dict['input_data']['node_t'][0]))
        for t in range(0, len(sic4dvar_dict['input_data']['node_t'][0])):
            test_t_array[t] = np.mean(sic4dvar_dict['input_data']['node_t'][:, t])
        sic4dvar_dict['input_data']['separate_reach_t'] = deepcopy(sic4dvar_dict['input_data']['reach_t'])
        sic4dvar_dict['input_data']['reach_t'] = deepcopy(test_t_array)
    sic4dvar_dict['output']['valid'] = 1
    sic4dvar_dict['output']['valid_a5'] = 1
    sic4dvar_dict['output']['valid_a5_sets'] = []
    if params.qsdev_activate:
        sic4dvar_dict['output']['q_std'] = input_data['reach_qsdev']
    sic4dvar_dict['output']['node_id'] = input_data['node_id']
    if params.force_specific_dates:
        logging.debug(f'reach_t: {sic4dvar_dict['input_data']['reach_t']}')
        dates = []
        for t in range(0, len(sic4dvar_dict['input_data']['reach_t'])):
            if check_na(sic4dvar_dict['input_data']['reach_t'][t]):
                dates.append(datetime.datetime(1, 1, 1))
            else:
                dates.append(seconds_to_date_old(sic4dvar_dict['input_data']['reach_t'][t]))
        indexes = [i for i, date in enumerate(dates) if params.start_date <= date <= params.end_date]
        if np.array(indexes).size == 0:
            if sic4dvar_dict['param_dict']['run_type'] == 'seq':
                logging.info('All data removed by date filtering.')
                logging.info("Sequential run, can't process without SWOT data.")
                sic4dvar_dict['output']['valid'] = 0
                return None
        else:
            reach_keys = ['reach_dA', 'reach_s', 'reach_w', 'reach_z', 'reach_t']
            node_keys = ['node_w', 'node_z', 'node_dA', 'node_s', 'node_t']
            for key in reach_keys:
                sic4dvar_dict['input_data'][key] = sic4dvar_dict['input_data'][key][indexes]
            for key in node_keys:
                sic4dvar_dict['input_data'][key] = sic4dvar_dict['input_data'][key][:, indexes]
            sic4dvar_dict['flag_dict']['nt'] = len(indexes)
            logging.debug(f'node_z shape: {sic4dvar_dict['input_data']['node_z'].shape}, reach_z shape: {sic4dvar_dict['input_data']['reach_z'].shape}')
    duplicate_t = np.ones(sic4dvar_dict['input_data']['node_t'].shape)
    for n in range(0, sic4dvar_dict['input_data']['node_t'].shape[0]):
        duplicate_t[n, :] = sic4dvar_dict['input_data']['reach_t']
    sic4dvar_dict['output']['reach_t_duplicate'] = deepcopy(duplicate_t)
    if sic4dvar_dict['input_data'] == 0:
        raise IOError('Error reading inputs ... input_data is empty')
    if 'node_t' in sic4dvar_dict['input_data'] and 'reach_w' in sic4dvar_dict['input_data']:
        if sic4dvar_dict['param_dict']['run_type'] == 'set':
            mask = np.ones(sic4dvar_dict['input_data']['node_t'].shape[1], dtype=bool)
            arr = np.full(sic4dvar_dict['input_data']['node_t'].shape[1], np.nan)
        if sic4dvar_dict['param_dict']['run_type'] == 'seq':
            mask = np.ones(len(sic4dvar_dict['input_data']['reach_w']), dtype=bool)
            arr = np.full(len(sic4dvar_dict['input_data']['reach_w']), np.nan)
        sic4dvar_dict['output']['q_algo5'] = np.ma.array(arr, mask=mask)
        sic4dvar_dict['output']['q_algo5_all'] = []
        sic4dvar_dict['output']['q_algo31'] = np.ma.array(arr, mask=mask)
        sic4dvar_dict['output']['a0'] = np.ma.array([np.nan], mask=[True])
        sic4dvar_dict['output']['n'] = np.ma.array([np.nan], mask=[True])
        sic4dvar_dict['output']['node_a0'] = []
        sic4dvar_dict['output']['node_n'] = []
        sic4dvar_dict['lim_node_xr'] = []
        sic4dvar_dict['output']['half_width'] = np.empty(len(sic4dvar_dict['input_data']['node_w']), dtype=object)
        sic4dvar_dict['output']['elevation'] = np.empty(len(sic4dvar_dict['input_data']['node_w']), dtype=object)
        sic4dvar_dict['output']['half_width'].fill(np.ma.array(arr, mask=mask))
        sic4dvar_dict['output']['elevation'].fill(np.ma.array(arr, mask=mask))
        sic4dvar_dict['output']['time'] = np.array([])
        sic4dvar_dict['removed_nodes_ind'] = np.zeros(0)
        sic4dvar_dict['indices'] = []
    elif sic4dvar_dict['param_dict']['run_type'] == 'seq':
        logging.info("ERROR: Sequential run, can't process without SWOT data.")
        return 0
    elif sic4dvar_dict['param_dict']['run_type'] == 'set':
        mask = []
        arr = []
        logging.info('INFO: Missing SWOT data. Set_run, processing.')
        sic4dvar_dict['output']['q_algo5'] = []
        sic4dvar_dict['output']['q_algo5_all'] = []
        sic4dvar_dict['output']['q_algo31'] = []
        sic4dvar_dict['output']['a0'] = []
        sic4dvar_dict['output']['n'] = []
        sic4dvar_dict['output']['node_a0'] = []
        sic4dvar_dict['output']['node_n'] = []
        sic4dvar_dict['lim_node_xr'] = []
        sic4dvar_dict['output']['half_width'] = []
        sic4dvar_dict['output']['elevation'] = []
        sic4dvar_dict['output']['time'] = np.array([])
        sic4dvar_dict['removed_nodes_ind'] = np.zeros(0)
        sic4dvar_dict['indices'] = []
    if not params.node_length:
        sic4dvar_dict['input_data']['node_x'] = sic4dvar_dict['input_data']['dist_out']
    if params.node_length:
        dist_out_1 = []
        if sic4dvar_dict['input_data']['dist_out'].size > 0:
            dist_out_1.append(float(sic4dvar_dict['input_data']['dist_out'][0]))
            if not params.start_from_downstream:
                direction = -1
            else:
                direction = 1
            for i in range(1, len(sic4dvar_dict['input_data']['node_length'])):
                dist_out_1.append(dist_out_1[i - 1] + direction * sic4dvar_dict['input_data']['node_length'][i - 1])
            dist_out_1 = dist_out_1[0] - np.array(dist_out_1)
            sic4dvar_dict['input_data']['node_x'] = np.array(dist_out_1)
        else:
            logging.info("No dist_out data available, can't reconstruct node_x based on node_length increments. Check your input data.")
            sic4dvar_dict['input_data']['node_x'] = np.array([])
    nb_valid_z = np.sum(~np.isnan(sic4dvar_dict['input_data']['node_z'].filled(np.nan)))
    nb_valid_w = np.sum(~np.isnan(sic4dvar_dict['input_data']['node_w'].filled(np.nan)))
    if nb_valid_z == 0 or nb_valid_w == 0:
        logging.debug(f'Not enough nodes with data to run SIC4DVAR: {nb_valid_z} valid WSE and {nb_valid_w} valid widths.')
        sic4dvar_dict['output']['valid'] = 0
        sic4dvar_dict['output']['stopped_stage'] = 'input_data_check'
        sic4dvar_dict['output']['node_z'] = sic4dvar_dict['input_data']['node_z']
        sic4dvar_dict['output']['node_w'] = sic4dvar_dict['input_data']['node_w']
    return sic4dvar_dict