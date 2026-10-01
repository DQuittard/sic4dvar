import logging
from copy import deepcopy
from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)
import numpy as np
import pandas as pd
import scipy
from pathlib import Path
import sic4dvar_params as params
from sic4dvar_functions.c550 import V, W
from sic4dvar_functions.sic4dvar_calculations import check_na, verify_name_length
from sic4dvar_functions.sic4dvar_helper_functions import compute_mean_var_from_2D_array, compute_mean_var_from_2D_array_sum, global_large_deviations_removal, grad_variance, global_large_deviations_removal_relative, global_large_deviations_removal_experimental
from sic4dvar_functions.sic4dvar_gnuplot_save import gnuplot_save, gnuplot_save_c1c2
from sic4dvar_functions.Y786 import K

def pooling(sic4dvar_dict, test_swot_z_obs):
    pass_nums = []
    for string in sic4dvar_dict['input_data']['pass_ids']:
        cycle, pass_num = string.split('_')
        pass_nums.append(int(pass_num))
    pass_nums = np.unique(pass_nums)
    nb_pools = len(pass_nums)
    pass_dict = {}
    for current_pass in pass_nums:
        total_sum = 0.0
        nb_valid_values = 0.0
        pass_dict[current_pass] = {}
        pass_dict[current_pass]['time_ids'] = []
        for string in sic4dvar_dict['input_data']['pass_ids']:
            id_t = np.argwhere(sic4dvar_dict['input_data']['pass_ids'] == string)[0][0]
            cycle, pass_num = string.split('_')
            if pass_num == str(current_pass):
                col_data = np.array(test_swot_z_obs[:, int(id_t)])
                total_sum += np.nansum(col_data)
                nb_valid_values += np.sum(~np.isnan(col_data))
                pass_dict[current_pass]['time_ids'].append(id_t)
        pass_dict[current_pass]['mean'] = total_sum / nb_valid_values if nb_valid_values > 0 else 0.0
        pass_dict[current_pass]['nb_valid_values'] = nb_valid_values
        pass_dict[current_pass]['weight'] = 1.0
    total_mean = 0.0
    total_scaling = 0.0
    for current_pass in pass_nums:
        total_mean += pass_dict[current_pass]['mean'] * pass_dict[current_pass]['weight'] * pass_dict[current_pass]['nb_valid_values']
        total_scaling += pass_dict[current_pass]['weight'] * pass_dict[current_pass]['nb_valid_values']
    total_mean = total_mean / total_scaling if total_scaling > 0 else np.nan
    for current_pass in pass_nums:
        for id_t in pass_dict[current_pass]['time_ids']:
            test_swot_z_obs[:, int(id_t)] = test_swot_z_obs[:, int(id_t)] + (total_mean - pass_dict[current_pass]['mean'])
    pass_dict_verif = {}
    return test_swot_z_obs

def Extrapolation(sic4dvar_dict, params):
    """
    Interpolates SWOT WSE observations in order to have all nodes
    observed at all SWOT time instants.
    Option 1 : Fortran version Uses weighted drops computed using
    the same pair of unknown and observed nodes from all overpasses.
    The drops (delta h between both nodes of tha same overpass) is
    weighted by the distance in time (delta h between the same node
    obeserved two different time instants).
    NOTE: This version involves a smoothing step before and after
    interpolation.

    Parameters
    ----------
    """
    sic4dvar_dict['input_data']['node_t_ini'] = deepcopy(sic4dvar_dict['input_data']['node_t'])
    sic4dvar_dict['input_data']['node_z_ini'] = deepcopy(sic4dvar_dict['input_data']['node_z'])
    sic4dvar_dict['input_data']['node_w_ini'] = deepcopy(sic4dvar_dict['input_data']['node_w'])
    node_x = sic4dvar_dict['input_data']['node_x']
    if params.corx_option == 0:
        corx = 200.0
    elif params.corx_option == 1:
        corx = np.mean(np.diff(node_x))
    elif params.corx_option == 2:
        corx = 200.0
    corx_array = np.ones(len(node_x)) * corx
    test_swot_z_obs = deepcopy(sic4dvar_dict['input_data']['node_z'])
    test_swot_w_obs = deepcopy(sic4dvar_dict['input_data']['node_w'])
    test_x_array = node_x
    test_t_array = sic4dvar_dict['input_data']['reach_t']
    cort = params.cort
    cort_width = cort
    cort_tmp = []
    cort_slope = 0.0
    for t in range(0, len(test_t_array)):
        if not check_na(test_t_array[t]):
            cort_tmp.append(test_t_array[t])
    cort_slope = np.mean(np.diff(cort_tmp))
    if params.override_cort:
        cort = cort_slope
    cort_wse = cort
    sic4dvar_dict['cort_wse'] = cort_wse
    sic4dvar_dict['cort_slope'] = 0.0
    sections_plot = False
    sic4dvar_dict['tmp_interp_values'] = []
    sic4dvar_dict['tmp_interp_values_w'] = []
    arr = sic4dvar_dict['input_data']['node_z_ini']
    arr[arr < -100000] = np.nan
    if sic4dvar_dict['param_dict']['gnuplot_saving']:
        reach_id = str(sic4dvar_dict['input_data']['reach_id'])
        reach_id = verify_name_length(reach_id)
        nodes2 = (node_x - node_x[0]) / 1000
        times2 = test_t_array / 3600 / 24
        output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', str(reach_id))
        if not Path(output_path).is_dir():
            Path(output_path).mkdir(parents=True, exist_ok=True)
        output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', str(reach_id), 'before_devia')
        gnuplot_save(nodes2, times2, test_swot_z_obs, sic4dvar_dict['input_data']['node_w_ini'], output_path, np.min(test_swot_z_obs), 2)
    reverse_order = False
    if params.large_deviations and sic4dvar_dict['param_dict']['run_type'] == 'seq':
        test_swot_z_obs, reverse_order = global_large_deviations_removal_experimental(sic4dvar_dict['input_data']['node_x'], sic4dvar_dict['input_data']['node_z_ini'], sic4dvar_dict['input_data']['reach_t'], times_debug=test_t_array)
        sic4dvar_dict['tmp_interp_values'].append(test_swot_z_obs)
        if sic4dvar_dict['param_dict']['gnuplot_saving']:
            nodes2 = (node_x - node_x[0]) / 1000
            times2 = test_t_array / 3600 / 24
            reach_id = str(sic4dvar_dict['input_data']['reach_id'])
            reach_id = verify_name_length(reach_id)
            output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', str(reach_id))
            if not Path(output_path).is_dir():
                Path(output_path).mkdir(parents=True, exist_ok=True)
            output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', str(reach_id), 'c1c2')
            output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', str(reach_id), 'after_devia')
            gnuplot_save(nodes2, times2, test_swot_z_obs, sic4dvar_dict['input_data']['node_w_ini'], output_path, np.min(test_swot_z_obs), 2)
    behavior = ''
    if reverse_order:
        if params.start_from_downstream:
            behavior = 'decrease'
        else:
            behavior = 'increase'
        logging.info('c1 > 0. : reverse order of nodes for interpolation')
    else:
        if params.start_from_downstream:
            behavior = 'increase'
        else:
            behavior = 'decrease'
        logging.info('c1 < 0. : normal order of nodes for interpolation')
    sic4dvar_dict['reverse_order'] = reverse_order
    if params.run_preprocessing:
        test_swot_z_obs = K(dim=1, value0_array=test_swot_z_obs, base0_array=test_x_array, max_iter=1, cor=corx_array, always_run_first_iter=True, behavior=behavior, inter_behavior=True, inter_behavior_min_thr=params.def_float_atol, inter_behavior_max_thr=params.DX_max_in, check_behavior='force', min_change_v_thr=0.0001, plot=False, plot_title='Relaxation sweep in space 1 WSE without Interchange', clean_run=True, debug_mode=False, time_integration=False)
    times = np.arange(0, len(sic4dvar_dict['input_data']['node_z'][0]))
    nodes = np.arange(0, len(sic4dvar_dict['input_data']['node_z']))
    times2 = np.around(test_t_array / 3600 / 24)
    times2 = times2 - min(times2)
    nodes2 = (node_x - node_x[0]) / 1000
    reach_id = str(sic4dvar_dict['input_data']['reach_id'])
    reach_id = verify_name_length(reach_id)
    if sic4dvar_dict['param_dict']['gnuplot_saving']:
        output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', reach_id)
        if not Path(output_path).is_dir():
            Path(output_path).mkdir(parents=True, exist_ok=True)
    sic4dvar_dict['tmp_interp_values'].append(test_swot_z_obs)
    if params.run_extrapolation:
        test_swot_z_obs = V(values0_array=test_swot_z_obs, space0_array=test_x_array, dx_max_in=params.DX_max_in, dx_max_out=params.DX_max_out, dw_min=0.01, clean_run=False, debug_mode=False, interp_missing_nodes=False)
    if sic4dvar_dict['param_dict']['gnuplot_saving']:
        reach_id = str(sic4dvar_dict['input_data']['reach_id'])
        reach_id = verify_name_length(reach_id)
        nodes2 = (node_x - node_x[0]) / 1000
        times2 = test_t_array / 3600 / 24
        output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', str(reach_id))
        if not Path(output_path).is_dir():
            Path(output_path).mkdir(parents=True, exist_ok=True)
        output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', str(reach_id), 'after_extrapolation')
        gnuplot_save(nodes2, times2, test_swot_z_obs, sic4dvar_dict['input_data']['node_w_ini'], output_path, np.min(test_swot_z_obs), 2)
    sic4dvar_dict['tmp_interp_values'].append(test_swot_z_obs)
    if params.run_preprocessing:
        for i in range(0, 1):
            test_swot_z_obs = K(dim=1, value0_array=test_swot_z_obs, base0_array=test_x_array, max_iter=10, cor=corx_array, always_run_first_iter=True, behavior=behavior, inter_behavior=True, inter_behavior_min_thr=params.def_float_atol, inter_behavior_max_thr=params.DX_max_in, check_behavior='force', min_change_v_thr=0.0001, plot=False, plot_title='Relaxation sweep in space 2 WSE With Interchange', clean_run=True, debug_mode=False, time_integration=False)
    sic4dvar_dict['tmp_interp_values'].append(test_swot_z_obs)
    if params.pooling:
        test_swot_z_obs = pooling(sic4dvar_dict, test_swot_z_obs)
    tmp = deepcopy(test_swot_z_obs)
    if params.run_preprocessing:
        test_swot_z_obs = K(dim=0, value0_array=test_swot_z_obs, base0_array=test_t_array, max_iter=1, cor=cort_wse, always_run_first_iter=False, behavior='', inter_behavior=False, inter_behavior_min_thr=params.def_float_atol, inter_behavior_max_thr=params.DX_max_in, check_behavior='', min_change_v_thr=0.0001, plot=False, plot_title='Relaxation sweep in time WSE without Interchange', clean_run=True, debug_mode=False, time_integration=False)
    sic4dvar_dict['tmp_interp_values'].append(test_swot_z_obs)
    reach_t = sic4dvar_dict['input_data']['reach_t']
    if params.run_preprocessing:
        test_swot_w_obs = K(dim=1, value0_array=test_swot_w_obs, base0_array=test_x_array, max_iter=1, cor=corx, always_run_first_iter=True, behavior='', inter_behavior=False, check_behavior='', min_change_v_thr=0.01, plot=False, plot_title='Relaxation sweep in space 1 W without Interchange', clean_run=True, debug_mode=False, time_integration=False)
    sic4dvar_dict['tmp_interp_values_w'].append(test_swot_w_obs)
    if params.run_extrapolation:
        test_swot_w_obs = W(values0_array=test_swot_w_obs, space0_array=test_x_array, weight0_array=test_swot_z_obs, dx_max_in=params.DX_max_in, dx_max_out=params.DX_max_out, dw_min=0.1, clean_run=False, debug_mode=False)
    sic4dvar_dict['tmp_interp_values_w'].append(test_swot_w_obs)
    if params.run_preprocessing:
        test_swot_w_obs = K(dim=1, value0_array=test_swot_w_obs, base0_array=test_x_array, max_iter=1, cor=corx, always_run_first_iter=True, behavior='', inter_behavior=False, check_behavior='', min_change_v_thr=0.01, plot=False, plot_title='Relaxation sweep in space 2 W without Interchange', clean_run=True, debug_mode=False, time_integration=False)
    sic4dvar_dict['tmp_interp_values_w'].append(test_swot_w_obs)
    if params.pooling:
        test_swot_w_obs = pooling(sic4dvar_dict, test_swot_w_obs)
    if params.run_preprocessing:
        test_swot_w_obs = K(dim=0, value0_array=test_swot_w_obs, base0_array=test_t_array, max_iter=1, cor=cort_wse, always_run_first_iter=True, behavior='', inter_behavior=False, check_behavior='', min_change_v_thr=0.01, plot=False, plot_title='Relaxation sweep in time W without Interchange', clean_run=True, debug_mode=False, time_integration=False)
    sic4dvar_dict['tmp_interp_values_w'].append(test_swot_w_obs)
    ' if not params.start_from_downstream:\n        test_swot_z_obs = test_swot_z_obs[::-1, :]\n        test_swot_w_obs = test_swot_w_obs[::-1, :]\n        test_x_array = test_x_array[::-1]\n        logging.info("Reversing order of nodes for interpolation and smoothing only") '
    sic4dvar_dict['input_data']['node_z'] = test_swot_z_obs.filled(np.nan)
    sic4dvar_dict['input_data']['node_w'] = test_swot_w_obs.filled(np.nan)
    return sic4dvar_dict