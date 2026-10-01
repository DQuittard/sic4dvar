import logging
from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)
import numpy as np
import pandas as pd
import scipy
import sic4dvar_params as params
from lib.lib_dates import get_swot_dates, seconds_to_date_old
from lib.lib_swot_obs import filter_swot_obs_on_quality, get_flag_dict_from_config
from sic4dvar_algos.X653 import *
from sic4dvar_algos.algo5 import *
from sic4dvar_functions.sic4dvar_calculations import compute_bb, verify_name_length
from sic4dvar_functions.sic4dvar_helper_functions import build_output_q_masked_array, correlation_nodes, get_weighted_q_data
from sic4dvar_modules.sic4dvar_compute_slope_and_bathymetry import bathymetry_computation, call_func_APR, slope_computation

def check_reach_data_SET(sic4dvar_dict, iR=0):
    """Returns true for if data has passed a check and false if not.
    """
    sic4dvar_dict['output']['q_algo31_masked'] = deepcopy(sic4dvar_dict['output']['q_algo31'][sic4dvar_dict['list_to_keep']])
    reach = dict()
    if sic4dvar_dict['param_dict']['run_type'] == 'seq':
        reach['dA'] = deepcopy([sic4dvar_dict['input_data']['reach_dA']])
        reach['w'] = deepcopy([sic4dvar_dict['input_data']['reach_w']])
        reach['s'] = deepcopy([sic4dvar_dict['input_data']['reach_s']])
        reach['z'] = deepcopy([sic4dvar_dict['input_data']['reach_z']])
    if sic4dvar_dict['param_dict']['run_type'] == 'set':
        reach['dA'] = deepcopy(sic4dvar_dict['filtered_data']['reach_dA'])
        reach['w'] = deepcopy(sic4dvar_dict['filtered_data']['reach_w'])
        reach['s'] = deepcopy(sic4dvar_dict['filtered_data']['reach_s'])
        reach['z'] = deepcopy(sic4dvar_dict['filtered_data']['reach_z'])

    def return_seq_filtered_data(dA, w, s):
        return (dA, w, s)

    def output(dA, w, s):
        if sic4dvar_dict['param_dict']['run_type'] == 'seq':
            sic4dvar_dict['filtered_data']['reach_dA'], sic4dvar_dict['filtered_data']['reach_w'], sic4dvar_dict['filtered_data']['reach_s'] = return_seq_filtered_data(reach['dA'][iR], reach['w'][iR], reach['s'][iR])
            sic4dvar_dict['filtered_data']['reach_dA'], sic4dvar_dict['filtered_data']['reach_w'], sic4dvar_dict['filtered_data']['reach_s'] = return_seq_filtered_data(reach['dA'][iR], reach['w'][iR], reach['s'][iR])
        if sic4dvar_dict['param_dict']['run_type'] == 'set':
            sic4dvar_dict['filtered_data']['reach_dA'][iR], sic4dvar_dict['filtered_data']['reach_w'][iR], sic4dvar_dict['filtered_data']['reach_s'][iR] = return_seq_filtered_data(reach['dA'][iR], reach['w'][iR], reach['s'][iR])
    if sic4dvar_dict['output']['q_algo31'].mask.all() or reach['dA'][iR].mask.all() or reach['w'][iR].mask.all() or reach['s'][iR].mask.all():
        return False
    elif sic4dvar_dict['output']['q_algo31'].mask.any() or reach['dA'][iR].mask.any() or reach['w'][iR].mask.any() or reach['s'][iR].mask.any():
        indices_to_mask = []
        keys = ['q_algo31', 'dA', 'w', 's']
        for key in keys:
            if key == 'q_algo31':
                data_mask = []
                for i in range(0, len(sic4dvar_dict['output'][key])):
                    data_mask.append(check_na(sic4dvar_dict['output'][key][i]))
            else:
                data_mask = []
                for i in range(0, len(reach[key][iR])):
                    data_mask.append(check_na(reach[key][iR][i]))
            index_to_mask = np.where(np.array(data_mask) == True)
            indices_to_mask.append(index_to_mask)
        indices_to_mask = np.concatenate([arr[0] for arr in indices_to_mask])
        indices_to_mask = np.unique(indices_to_mask)
        sic4dvar_dict['removed_indices'] = indices_to_mask
        if indices_to_mask.size > 0:
            mask = np.ones(len(sic4dvar_dict['output']['q_algo31']), dtype=bool)
            mask[indices_to_mask] = False
            sic4dvar_dict['output']['q_algo31_masked'] = sic4dvar_dict['output']['q_algo31'][mask]
            reach['dA'][iR] = reach['dA'][iR][mask]
            reach['w'][iR] = reach['w'][iR][mask]
            reach['s'][iR] = reach['s'][iR][mask]
            keys = ['dA', 'w', 's', 'q_algo31_masked']
            for key in keys:
                full_nan = 0
                for t in range(0, len(sic4dvar_dict['output']['q_algo31_masked'])):
                    if not key == 'q_algo31_masked':
                        if check_na(reach[key][iR][t]):
                            full_nan += 1
                    elif check_na(sic4dvar_dict['output'][key][t]):
                        full_nan += 1
                if full_nan == len(sic4dvar_dict['output']['q_algo31_masked']):
                    output(reach['dA'][iR], reach['w'][iR], reach['s'][iR])
                    return False
        if len(sic4dvar_dict['output']['q_algo31_masked']) < sic4dvar_dict['param_dict']['min_obs']:
            logging.info('Not enough timestamps to process with algo5!')
            output(reach['dA'][iR], reach['w'][iR], reach['s'][iR])
            return False
    OUTLOG = []
    for ind in range(len(reach['w'][iR])):
        "\n        if params.cython_version:\n            if not cy.check_suitability(abs(sic4dvar_dict['filtered_data']['reach_dA'][iR].min())*1.5,sic4dvar_dict['filtered_data']['reach_dA'][iR][ind], sic4dvar_dict['filtered_data']['reach_s'][iR][ind], 0.1, sic4dvar_dict['filtered_data']['reach_w'][iR][ind]):\n                OUTLOG += [True]\n            else:\n                OUTLOG +=[False]\n        "
        if not calc.check_suitability(0.1, abs(reach['dA'][iR].min()) * 1.5, reach['dA'][iR][ind], reach['w'][iR][ind], reach['s'][iR][ind]):
            OUTLOG += [True]
        else:
            OUTLOG += [False]
    if np.array(OUTLOG).all():
        logging.info('Input data into algo5 is not suitable for algorithm!')
        output(reach['dA'][iR], reach['w'][iR], reach['s'][iR])
        return False
    else:
        output(reach['dA'][iR], reach['w'][iR], reach['s'][iR])
        return True
    output(reach['dA'][iR], reach['w'][iR], reach['s'][iR])
    return True

def optional_dA_fill(sic4dvar_dict, input_reach_dA, filtered_reach_w, filtered_reach_z, params):
    count = 0
    for t in range(0, len(input_reach_dA)):
        if check_na(input_reach_dA[t]):
            count += 1
    if count == len(input_reach_dA):
        logging.info('All input reach dA values are missing.')
    elif params.force_compute:
        logging.info('Force compute is enabled.')
    if count == len(input_reach_dA) or params.force_compute:
        logging.info('Computing dA.')
        if not params.dA_from_mike_method:
            logging.info('Computing dA using sic4dvar bathymetry computation.')
            a5_xr, a5_yr, _ = bathymetry_computation([filtered_reach_w], [filtered_reach_z], sic4dvar_dict['param_dict'], params, input_data=sic4dvar_dict['input_data'], filtered_data=sic4dvar_dict['filtered_data'], slope=sic4dvar_dict['input_data']['reach_s'], algo='algo5')
            node_a, _, _, _, _ = call_func_APR([filtered_reach_w], [filtered_reach_z], a5_xr, a5_yr, params, sic4dvar_dict['param_dict'])
            masked_data = np.ma.masked_values(np.array(node_a[0]), value=-9999.0)
            filtered_reach_dA = masked_data
        elif params.dA_from_mike_method:
            logging.info('Computing dA using Mike method.')
            a5_xr, a5_yr, node_a = bathymetry_computation([filtered_reach_w], [filtered_reach_z], sic4dvar_dict['param_dict'], params, input_data=sic4dvar_dict['input_data'], filtered_data=sic4dvar_dict['filtered_data'], slope=sic4dvar_dict['input_data']['reach_s'], force_method='Mike', algo='algo5')
            masked_data = np.ma.masked_values(np.array(node_a), value=-9999.0)
            filtered_reach_dA = masked_data
        return filtered_reach_dA
    elif not count == len(input_reach_dA) and (not params.force_compute):
        logging.info('Using existing input reach dA values.')
        return input_reach_dA

def algo5_set_error_state(sic4dvar_dict, i=0):
    if sic4dvar_dict['param_dict']['run_type'] == 'seq':
        sic4dvar_dict['output']['valid_a5'] = 0
    if sic4dvar_dict['param_dict']['run_type'] == 'set':
        sic4dvar_dict['output']['valid_a5_sets'][i] = 0

def prepare_and_verify_reach_data(sic4dvar_dict, use_aggregated_node_data=False, use_projected_width=False, i=0):
    reach_z = deepcopy(sic4dvar_dict['input_data']['reach_z'])
    reach_w = deepcopy(sic4dvar_dict['input_data']['reach_w'])
    count_z = 0
    count_w = 0
    for t in range(0, len(sic4dvar_dict['input_data']['reach_z'])):
        if check_na(sic4dvar_dict['input_data']['reach_z'][t]):
            count_z += 1
        if check_na(sic4dvar_dict['input_data']['reach_w'][t]):
            count_w += 1
    count_reach_xr = 0
    count_reach_yr = 0
    for t in range(0, len(sic4dvar_dict['input_data']['reach_xr'])):
        if check_na(sic4dvar_dict['input_data']['reach_xr'][t]):
            count_reach_xr += 1
        if check_na(sic4dvar_dict['input_data']['reach_yr'][t]):
            count_reach_yr += 1
    if count_reach_xr >= len(sic4dvar_dict['input_data']['reach_xr']) or count_reach_yr >= len(sic4dvar_dict['input_data']['reach_yr']):
        logging.warning('ALGO 5: Reach bathymetry data is missing/all NaNs for width or elevation, need more than 0 point to compute.')
        algo5_set_error_state(sic4dvar_dict, i)
        return (np.full(sic4dvar_dict['input_data']['reach_w'].shape, np.nan), np.full(sic4dvar_dict['input_data']['reach_z'].shape, np.nan))
    if count_z >= len(sic4dvar_dict['input_data']['reach_z']) - 1:
        logging.warning('ALGO 5: Reach elevation data is missing, need more than 1 time to compute.')
        algo5_set_error_state(sic4dvar_dict, i)
        return (np.full(sic4dvar_dict['input_data']['reach_w'].shape, np.nan), np.full(sic4dvar_dict['input_data']['reach_z'].shape, np.nan))
    if count_w >= len(sic4dvar_dict['input_data']['reach_w']) - 1:
        logging.warning('ALGO 5: Reach width data is missing, need more than 1 time to compute.')
        if use_projected_width:
            logging.warning('ALGO 5: Using projected width to fill missing reach width data.')
        else:
            algo5_set_error_state(sic4dvar_dict, i)
            return (np.full(sic4dvar_dict['input_data']['reach_w'].shape, np.nan), np.full(sic4dvar_dict['input_data']['reach_z'].shape, np.nan))
    if use_projected_width:
        logging.warning('ALGO 5: Using projected width.')
        reach_w_projected = np.full(sic4dvar_dict['input_data']['reach_z'].shape, np.nan)
        valid = sic4dvar_dict['input_data']['reach_z'] > -100000000000.0
        reach_w_projected[valid] = np.interp(sic4dvar_dict['input_data']['reach_z'][valid], sic4dvar_dict['input_data']['reach_yr'], sic4dvar_dict['input_data']['reach_xr'])
        reach_w = deepcopy(reach_w_projected)
        masked_data = np.ma.masked_values(np.array(reach_w), value=-9999.0)
        reach_w = masked_data
    return (reach_w, reach_z)

def launch_algo5(sic4dvar_dict, NBR_REACHES):
    for i in range(0, NBR_REACHES):
        if sic4dvar_dict['param_dict']['run_type'] == 'seq':
            sic4dvar_dict['input_data']['reach_w'], sic4dvar_dict['input_data']['reach_z'] = prepare_and_verify_reach_data(sic4dvar_dict, params.use_aggregated_node_data, params.use_projected_width)
            if sic4dvar_dict['output']['valid_a5']:
                sic4dvar_dict['input_data']['reach_dA_ini'] = deepcopy(sic4dvar_dict['input_data']['reach_dA'])
                sic4dvar_dict['input_data']['reach_dA'] = optional_dA_fill(sic4dvar_dict, sic4dvar_dict['input_data']['reach_dA'], sic4dvar_dict['input_data']['reach_w'], sic4dvar_dict['input_data']['reach_z'], params)
                sic4dvar_dict['reach_check'] = check_reach_data_SET(sic4dvar_dict)
            else:
                logging.warning('ALGO 5: No valid reach w or z. Cannot run algo5.')
                sic4dvar_dict['reach_check'] = False
        if sic4dvar_dict['param_dict']['run_type'] == 'set':
            sic4dvar_dict['filtered_data']['reach_dA'][i] = optional_dA_fill(sic4dvar_dict, sic4dvar_dict['input_data']['reach_dA'][i], sic4dvar_dict['filtered_data']['reach_w'][i], sic4dvar_dict['filtered_data']['reach_z'][i], params)
            sic4dvar_dict['reach_check'] = check_reach_data_SET(sic4dvar_dict, i)
        if sic4dvar_dict['reach_check']:
            logging.info('Running algo5 at REACH level to estimate discharge, A0 & n.')
            if sic4dvar_dict['param_dict']['run_type'] == 'seq':
                t0_a5 = datetime.utcnow()
                import matplotlib.pyplot as plt
                sic4dvar_dict['output']['q_algo5'], sic4dvar_dict['algo5_results'] = algo5(sic4dvar_dict['output']['q_algo31_masked'], sic4dvar_dict['filtered_data']['reach_dA'], sic4dvar_dict['filtered_data']['reach_w'], sic4dvar_dict['filtered_data']['reach_s'], sic4dvar_dict['input_data']['reach_id'], equation='ManningLW')
                t1_a5 = datetime.utcnow()
                sic4dvar_dict['output']['q_algo5'] = algo5_fill_removed_data(sic4dvar_dict['output']['q_algo5'], sic4dvar_dict['removed_indices'], len(sic4dvar_dict['input_data']['node_z'][0]))
            if sic4dvar_dict['param_dict']['run_type'] == 'set':
                t0_a5 = datetime.utcnow()
                sic4dvar_dict['output']['q_algo5'], sic4dvar_dict['algo5_results'] = algo5(sic4dvar_dict['output']['q_algo31_masked'], sic4dvar_dict['filtered_data']['reach_dA'][i], sic4dvar_dict['filtered_data']['reach_w'][i], sic4dvar_dict['filtered_data']['reach_s'][i], sic4dvar_dict['input_data']['reach_id'][i], equation='ManningLW')
                t1_a5 = datetime.utcnow()
                sic4dvar_dict['output']['q_algo5'] = algo5_fill_removed_data(sic4dvar_dict['output']['q_algo5'], sic4dvar_dict['removed_indices'], len(sic4dvar_dict['filtered_data']['node_z'][0]))
                sic4dvar_dict['output']['q_algo5'] = build_output_q_masked_array(sic4dvar_dict, 'q_algo5')
                sic4dvar_dict['output']['q_algo5_all'] += [sic4dvar_dict['output']['q_algo5']]
        else:
            algo5_set_error_state(sic4dvar_dict, i)
    return sic4dvar_dict