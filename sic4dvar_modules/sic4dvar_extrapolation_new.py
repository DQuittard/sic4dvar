import logging
from copy import deepcopy
from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)
import numpy as np
from pathlib import Path
from sic4dvar_functions.c550 import V, W
from sic4dvar_functions.sic4dvar_calculations import verify_name_length
from sic4dvar_functions.sic4dvar_helper_functions import global_large_deviations_removal_experimental
from sic4dvar_functions.sic4dvar_gnuplot_save import gnuplot_save
from sic4dvar_functions.Y786 import K

def igor_method(node_z, corx, node_x, LSMX, eps1, eps2):
    k_T1 = node_z.shape[1]
    lmax = node_z.shape[0]
    rel0 = np.zeros(lmax)
    orig_z = deepcopy(node_z)
    for ist in range(0, k_T1):
        ss = 0.0
        for n in range(0, lmax):
            ss = ss + node_z[n, ist]
        ZMEAN0 = ss / lmax
        ss0 = 0.0
        for n in range(0, lmax):
            ss0 = ss0 + (node_z[n, ist] - ZMEAN0) ** 2
        ss0 = np.sqrt(ss0 / lmax)
        for n in range(1, lmax):
            rel0[n] = 1.0 - np.exp(-1.0 * ((node_x[n] - node_x[n - 1]) / corx))
        ss1 = 0.0
        for it in range(0, LSMX):
            i_reo = 0
            for n in range(lmax - 2, 0, -1):
                node_z[n, ist] = node_z[n + 1, ist] * (1.0 - rel0[n + 1]) + node_z[n, ist] * rel0[n + 1]
                if it > 0:
                    if node_z[n, ist] < node_z[n + 1, ist] - eps1:
                        ss = node_z[n, ist]
                        node_z[n, ist] = node_z[n + 1, ist]
                        node_z[n + 1, ist] = ss
                        i_reo = i_reo + 1
            for n in range(1, lmax):
                node_z[n, ist] = node_z[n - 1, ist] * (1.0 - rel0[n]) + node_z[n, ist] * rel0[n]
                if it > 0:
                    if node_z[n, ist] > node_z[n - 1, ist] + eps1:
                        ss = node_z[n - 1, ist]
                        node_z[n - 1, ist] = node_z[n, ist]
                        node_z[n, ist] = ss
                        i_reo = i_reo + 1
            ss = 0.0
            for n in range(0, lmax):
                ss = ss + (node_z[n, ist] - orig_z[n, ist]) ** 2
            ss = np.sqrt(ss / lmax)
            if it > 0:
                if np.abs((ss - ss1) / ss0) > eps2:
                    ss1 = ss
                else:
                    break
            else:
                ss1 = ss
        ss = 0.0
        for n in range(0, lmax):
            ss = ss + node_z[n, ist]
        ZMEAN1 = ss / lmax
        for n in range(0, lmax):
            node_z[n, ist] = node_z[n, ist] + ZMEAN0 - ZMEAN1
    return node_z

def pooling(sic4dvar_dict, node_z):
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
                col_data = np.array(node_z[:, int(id_t)])
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
            node_z[:, int(id_t)] = node_z[:, int(id_t)] + (total_mean - pass_dict[current_pass]['mean'])
    pass_dict_verif = {}
    return node_z

def nan_substitution_elevation(array, cslope, node_x):
    from lib.lib_verif import check_na
    i_vl = -1
    i_vr2 = -1
    i_vr = -1
    new_array = deepcopy(array)
    for n in range(0, len(array)):
        if not check_na(array[n]):
            i_vl = n
            new_array[n] = array[n]
        else:
            if n > i_vr:
                i_vr1 = -1
                if i_vr2 != 0:
                    for n1 in range(n + 1, len(array)):
                        if not check_na(array[n1]):
                            i_vr1 = n1
                            break
                        else:
                            pass
            i_vr = i_vr1
            if i_vr1 == -1:
                i_vr2 = 0
            if i_vl != -1 and i_vr != -1:
                new_array[n] = array[i_vl] + (array[i_vr] - array[i_vl]) * (node_x[n] - node_x[i_vl]) / (node_x[i_vr] - node_x[i_vl])
                logging.info(f'Interpolating node {n} between valid nodes {i_vl} and {i_vr}')
            elif i_vl == -1 and i_vr != -1:
                new_array[n] = array[i_vr] + cslope * (node_x[n] - node_x[i_vr])
                logging.info(f'Extrapolating to the left node {n} using valid node {i_vr}')
            elif i_vl != -1 and i_vr == -1:
                new_array[n] = array[i_vl] + cslope * (node_x[n] - node_x[i_vl])
                logging.info(f'Extrapolating to the right node {n} using valid node {i_vl}')
            elif i_vl == -1 and i_vr == -1:
                new_array[n] = np.nan
                logging.warning(f'Cannot interpolate or extrapolate node {n} because no valid nodes found.')
    return new_array

def Extrapolation(node_z, node_w, node_x, reach_t, corx, corx_array, cort_wse, cort_width, gnuplot_saving=False, reach_id=None, output_dir=None, run_type='seq', use_large_deviations=True, start_from_downstream=False, run_preprocessing=True, run_extrapolation=True, pooling=False, regularization_profile=False, use_quantiles_for_densification=False):
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
    node_z_ini = deepcopy(node_z)
    optionnal_output_dict = {}
    optionnal_output_dict['tmp_interp_values'] = []
    optionnal_output_dict['tmp_interp_values_w'] = []
    if gnuplot_saving:
        reach_id = str(reach_id)
        reach_id = verify_name_length(reach_id)
        nodes2 = (node_x - node_x[0]) / 1000
        times2 = reach_t / 3600 / 24
        output_path = Path(output_dir).joinpath('gnuplot_data', str(reach_id))
        if not output_path.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir).joinpath('gnuplot_data', str(reach_id), 'before_devia')
        gnuplot_save(nodes2, times2, node_z, node_w, output_path, np.min(node_z), 2)
    reverse_order = False
    if use_large_deviations and run_type == 'seq':
        node_z, reverse_order, c1, c2 = global_large_deviations_removal_experimental(node_x, node_z, reach_t, times_debug=reach_t)
        node_z_devia = deepcopy(node_z)
        optionnal_output_dict['tmp_interp_values'].append(node_z)
        if gnuplot_saving:
            nodes2 = (node_x - node_x[0]) / 1000
            times2 = reach_t / 3600 / 24
            reach_id = str(reach_id)
            reach_id = verify_name_length(reach_id)
            output_path = Path(output_dir).joinpath('gnuplot_data', str(reach_id))
            if not Path(output_path).is_dir():
                Path(output_path).mkdir(parents=True, exist_ok=True)
            output_path = Path(output_dir).joinpath('gnuplot_data', str(reach_id), 'c1c2')
            output_path = Path(output_dir).joinpath('gnuplot_data', str(reach_id), 'after_devia')
            gnuplot_save(nodes2, times2, node_z, node_w, output_path, np.min(node_z), 2)
    behavior = ''
    if reverse_order:
        if start_from_downstream:
            behavior = 'decrease'
        else:
            behavior = 'increase'
        logging.info('c1 > 0. : reverse order of nodes for interpolation')
    else:
        if start_from_downstream:
            behavior = 'increase'
        else:
            behavior = 'decrease'
        logging.info('c1 < 0. : normal order of nodes for interpolation')
    optionnal_output_dict['reverse_order'] = reverse_order
    if run_preprocessing:
        node_z = K(dim=1, value0_array=node_z, base0_array=node_x, max_iter=1, cor=corx_array, always_run_first_iter=True, behavior=behavior, inter_behavior=True, inter_behavior_min_thr=0.01, inter_behavior_max_thr=np.inf, check_behavior='force', min_change_v_thr=0.0001, plot=False, plot_title='Relaxation sweep in space 1 WSE without Interchange', clean_run=True, debug_mode=False, time_integration=False)
    if gnuplot_saving:
        times = np.arange(0, len(node_z[0]))
        nodes = np.arange(0, len(node_z))
        times2 = np.around(reach_t / 3600 / 24)
        times2 = times2 - min(times2)
        nodes2 = (node_x - node_x[0]) / 1000
        reach_id = str(reach_id)
        reach_id = verify_name_length(reach_id)
        output_path = Path(output_dir).joinpath('gnuplot_data', reach_id)
        if not Path(output_path).is_dir():
            Path(output_path).mkdir(parents=True, exist_ok=True)
    optionnal_output_dict['tmp_interp_values'].append(node_z)
    if regularization_profile:
        if c1 == 0.0:
            c1 = -1e-06
        else:
            c1 = c1 * 0.001
        if not use_quantiles_for_densification:
            logging.info('Using mean elevation profile for densification')
            from sic4dvar_functions.sic4dvar_helper_functions import compute_mean_elevation_profile
            mean_elevation_profile = compute_mean_elevation_profile(node_z, reach_t)
            new_mean_elevation_profile = nan_substitution_elevation(mean_elevation_profile, c1, node_x)
            c1 = 0.0
            mean_width_profile = compute_mean_elevation_profile(node_w, reach_t)
            new_mean_width_profile = nan_substitution_elevation(mean_width_profile, c1, node_x)
            node_z[:, -1] = new_mean_elevation_profile
            node_w[:, -1] = new_mean_width_profile
        else:
            logging.info('Using quantiles for densification')
            quantile_matrix0 = np.zeros((node_z.shape[0], 3))
            for n in range(0, node_z.shape[0]):
                quantile_matrix0[n, :] = np.nanquantile(node_z[n, :], [0.33, 0.66, 0.99])
            for nb_quant in range(0, 3):
                quantile_matrix0[:, nb_quant] = nan_substitution_elevation(quantile_matrix0[:, nb_quant], c1, node_x)
                node_z[:, -1 - nb_quant] = quantile_matrix0[:, nb_quant]
    if False:
        mean_elevation_profile_array = np.ones(node_z.shape[0]) * np.nan
        for n in range(0, node_z.shape[0]):
            array = node_z[n, :]
            dimension = reach_t
            array_mean = 0.0
            time_scaling = 0.0
            stop = 0
            from lib.lib_verif import check_na
            for t in range(1, len(array)):
                if not check_na(array[t - 1]) and (not check_na(dimension[t - 1])):
                    for t1 in range(t, len(array)):
                        if not check_na(array[t1]) and (not check_na(dimension[t1])):
                            array_mean += (array[t1] + array[t - 1]) / 2 * (dimension[t1] - dimension[t - 1])
                            time_scaling += dimension[t1] - dimension[t - 1]
                        elif t1 == len(array) - 1 and array_mean == 0.0 and (time_scaling == 0.0):
                            stop = 1
                            break
                        else:
                            continue
            if stop == 1:
                logging.error(f'Non integrable sequence for section {n}')
                continue
            if time_scaling == 0:
                array_mean = np.nan
            else:
                array_mean = array_mean / time_scaling
            if array_mean == 0:
                array_mean = np.nan
            mean_elevation_profile_array[n] = array_mean
        if np.all(mean_elevation_profile_array == 0):
            logging.error(f'Mean elevation profile is zero for all sections. Check.')
        print('Mean elevation profile array:', mean_elevation_profile_array)
        print(bug)
    node_z_in_between = deepcopy(node_z)
    if run_extrapolation:
        node_z = V(values0_array=node_z, space0_array=node_x, dx_max_in=np.inf, dx_max_out=500, dw_min=0.01, clean_run=False, debug_mode=False, interp_missing_nodes=True)
    if gnuplot_saving:
        reach_id = str(reach_id)
        reach_id = verify_name_length(reach_id)
        nodes2 = (node_x - node_x[0]) / 1000
        times2 = reach_t / 3600 / 24
        output_path = Path(output_dir).joinpath('gnuplot_data', str(reach_id))
        if not Path(output_path).is_dir():
            Path(output_path).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir).joinpath('gnuplot_data', str(reach_id), 'after_extrapolation')
        gnuplot_save(nodes2, times2, node_z, node_w, output_path, np.min(node_z), 2)
    optionnal_output_dict['tmp_interp_values'].append(node_z)
    if run_preprocessing:
        for i in range(0, 1):
            node_z = K(dim=1, value0_array=node_z, base0_array=node_x, max_iter=10, cor=corx_array, always_run_first_iter=True, behavior=behavior, inter_behavior=True, inter_behavior_min_thr=0.01, inter_behavior_max_thr=500, check_behavior='force', min_change_v_thr=0.0001, plot=False, plot_title='Relaxation sweep in space 2 WSE With Interchange', clean_run=True, debug_mode=False, time_integration=False)
    optionnal_output_dict['tmp_interp_values'].append(node_z)
    if pooling:
        node_z = pooling(sic4dvar_dict, node_z)
    tmp = deepcopy(node_z)
    if run_preprocessing:
        node_z = K(dim=0, value0_array=node_z, base0_array=reach_t, max_iter=1, cor=cort_wse, always_run_first_iter=False, behavior='', inter_behavior=False, inter_behavior_min_thr=0.01, inter_behavior_max_thr=500, check_behavior='', min_change_v_thr=0.0001, plot=False, plot_title='Relaxation sweep in time WSE without Interchange', clean_run=True, debug_mode=False, time_integration=False)
    optionnal_output_dict['tmp_interp_values'].append(node_z)
    if run_preprocessing:
        node_w = K(dim=1, value0_array=node_w, base0_array=node_x, max_iter=1, cor=corx, always_run_first_iter=True, behavior='', inter_behavior=False, check_behavior='', min_change_v_thr=0.01, plot=False, plot_title='Relaxation sweep in space 1 W without Interchange', clean_run=True, debug_mode=False, time_integration=False)
    optionnal_output_dict['tmp_interp_values_w'].append(node_w)
    if run_extrapolation:
        node_w = W(values0_array=node_w, space0_array=node_x, weight0_array=node_z, dx_max_in=np.inf, dx_max_out=500, dw_min=0.1, clean_run=False, debug_mode=False)
    optionnal_output_dict['tmp_interp_values_w'].append(node_w)
    if run_preprocessing:
        node_w = K(dim=1, value0_array=node_w, base0_array=node_x, max_iter=1, cor=corx, always_run_first_iter=True, behavior='', inter_behavior=False, check_behavior='', min_change_v_thr=0.01, plot=False, plot_title='Relaxation sweep in space 2 W without Interchange', clean_run=True, debug_mode=False, time_integration=False)
    optionnal_output_dict['tmp_interp_values_w'].append(node_w)
    if pooling:
        node_w = pooling(sic4dvar_dict, node_w)
    if run_preprocessing:
        node_w = K(dim=0, value0_array=node_w, base0_array=reach_t, max_iter=1, cor=cort_wse, always_run_first_iter=True, behavior='', inter_behavior=False, check_behavior='', min_change_v_thr=0.01, plot=False, plot_title='Relaxation sweep in time W without Interchange', clean_run=True, debug_mode=False, time_integration=False)
    optionnal_output_dict['tmp_interp_values_w'].append(node_w)
    ' if not start_from_downstream:\n        node_z = node_z[::-1, :]\n        node_w = node_w[::-1, :]\n        node_x = node_x[::-1]\n        logging.info("Reversing order of nodes for interpolation and smoothing only") '
    return (np.ma.filled(node_z, np.nan), np.ma.filled(node_w, np.nan), optionnal_output_dict)