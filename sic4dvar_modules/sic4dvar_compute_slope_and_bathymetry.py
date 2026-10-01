from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)
import pandas as pd
import scipy
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from copy import deepcopy
import sic4dvar_params as params
from sic4dvar_functions.sic4dvar_gnuplot_save import gnuplot_save, gnuplot_save_list, gnuplot_save_slope
from sic4dvar_functions.sic4dvar_calculations import check_na, verify_name_length, compute_bb, fnc_APR, f_approx_sections_v6
from sic4dvar_functions.Y786 import K
from sic4dvar_functions.cs.G546 import M
try:
    from Confluence.input.input.extract.CalculateHWS import CalculateHWS
    from Confluence.input.input.extract.DomainHWS import DomainHWS
    from Confluence.input.input.extract.HWS_IO import HWS_IO
    Confluence_HWS_method = True
except ImportError:
    Confluence_HWS_method = False

def create_confluence_dict(reach_time, reach_z, reach_w, reach_s):
    ObsData = {}
    ObsData['nR'] = 1
    ObsData['xkm'] = np.nan
    ObsData['L'] = np.nan
    ObsData['nt'] = len(reach_time)
    '\n    ts = reach_time.data #swot_dataset["time"].values\n    epoch = datetime.datetime(2000,1,1,0,0,0)\n    tall = []\n    for t in ts:\n        if t > 0:\n            delta = epoch + datetime.timedelta(seconds=t)\n            tall.append(delta.timestamp())\n        else:\n            tall.append(0)\n   \n    # tall = [ epoch + datetime.timedelta(seconds=t) if t > 0 else 0 for t in ts]\n    '
    ObsData['t'] = reach_time
    ObsData['h'] = np.empty((ObsData['nR'], ObsData['nt']))
    ObsData['h0'] = np.empty((ObsData['nR'], 1))
    ObsData['S'] = np.empty((ObsData['nR'], ObsData['nt']))
    ObsData['w'] = np.empty((ObsData['nR'], ObsData['nt']))
    for i in range(0, ObsData['nR']):
        ObsData['h'][i, :] = reach_z
        ObsData['w'][i, :] = reach_w
        ObsData['S'][i, :] = reach_s
    ObsData['sigh'] = 0.1
    ObsData['sigw'] = 10.0
    ObsData['sigS'] = 1.7e-05
    ObsData['iDelete'] = np.where(np.isnan(ObsData['w'][0, :]) | np.isnan(ObsData['h'][0, :]))
    "\n    iUse = logical_not(isnan(ObsData['w'][0,:]) |                         isnan(ObsData['h'][0,:]))\n\n    self.SubSelectData(iUse)\n    "
    return ObsData

def call_func_APR(node_w, node_z, node_xr, node_yr, params, param_dict, coeff_array=[], t=-1.0):
    node_a = node_w.copy()
    node_p = node_w.copy()
    node_r = node_w.copy()
    node_w_simp = node_w.copy()
    depth_mean = []
    A0_mean = []
    P0_mean = []
    W0_mean = []
    if np.array(coeff_array).size < 1.0:
        coeff_array = np.ones(len(node_z[0]))
    for i in range(len(node_w)):
        if t >= 0.0:
            node_a[i], node_p[i], node_r[i], node_w_simp[i] = fnc_APR(node_z[i, t] * coeff_array, node_xr[i], node_yr[i])
        else:
            node_a[i], node_p[i], node_r[i], node_w_simp[i] = fnc_APR(node_z[i] * coeff_array, node_xr[i], node_yr[i])
        depth_section = 0.0
        if param_dict['bb_computation']:
            depth_section = np.nanmax(node_yr[i]) - np.nanmin(node_yr[i])
            A0 = 0
            P0 = node_xr[i][0]
            for j in range(0, len(node_xr[i]) - 1):
                A0 = A0 + (node_xr[i][j] + node_xr[i][j + 1]) / 2 * (node_yr[i][j + 1] - node_yr[i][j])
                P0 = P0 + 2 * np.sqrt((node_xr[i][j + 1] / 2 - node_xr[i][j] / 2) ** 2 + (node_yr[i][j + 1] - node_yr[i][j]) ** 2)
                if node_yr[i][j] - np.nanmin(node_yr[i]) > (np.nanmax(node_yr[i]) - np.nanmin(node_yr[i])) / 2:
                    depth_section = (np.nanmax(node_yr[i]) - np.nanmin(node_yr[i])) / 2
                    break
            if False:
                results2 = f_approx_sections_v6(node_w[i], node_z[i], params.approx_section_params[0], params.approx_section_params[1], params.approx_section_params[2])
                plt.plot(node_xr[i], node_yr[i])
                plt.plot(results2[0], results2[1])
                plt.plot(node_w[i], node_z[i], marker='.', linestyle='None')
                print('A0, P0, R, depth, depth/R:', A0, P0, A0 / P0, depth_section, depth_section / (A0 / P0))
                plt.show()
                plt.clf()
            A0_mean.append(A0)
            P0_mean.append(P0)
            W0_mean.append(node_xr[i][0])
        index_temp = np.where(node_w_simp[i] == 0.0)
        node_w_simp[i][index_temp] = np.nanmin(node_xr[i][np.where(node_xr[i] > 0.0)])
        depth_mean.append(depth_section)
    if not param_dict['bb_computation']:
        bb = 9999.0
    else:
        depth_mean = np.nanmean(depth_mean)
        A0_mean = np.nanmean(A0_mean)
        P0_mean = np.nanmean(P0_mean)
        W0_mean = np.nanmean(W0_mean)
        print('depth_mean, A0_mean, P0_mean, W0_mean:', depth_mean, A0_mean, P0_mean, W0_mean)
        print('mean Radius:', A0_mean / P0_mean, depth_mean / (A0_mean / P0_mean))
        bb = compute_bb(depth_mean, W0_mean, A0_mean, P0_mean)
        print('Final computed bb:', bb)
        if bb < 1:
            bb = 1
        print('Final bb:', bb)
    return (node_a, node_p, node_r, node_w_simp, bb)

def slope_computation(sic4dvar_dict):
    if sic4dvar_dict['param_dict']['use_reach_slope']:
        if sic4dvar_dict['param_dict']['run_type'] == 'set':
            for t in range(sic4dvar_dict['filtered_data']['node_z'][0].shape[0]):
                SLOPEM1 = np.zeros(sic4dvar_dict['filtered_data']['node_z'][0].shape[0])
                if sic4dvar_dict['filtered_data']['reach_s'][0][t] > 0.0:
                    if not params.node_length:
                        SLOPEM1[t] = sic4dvar_dict['filtered_data']['reach_s'][0][t] * (sic4dvar_dict['filtered_data']['node_x'][-1] - sic4dvar_dict['filtered_data']['node_x'][0])
                        for i in range(1, sic4dvar_dict['filtered_data']['node_x'].size):
                            pass
                    elif params.node_length:
                        SLOPEM1[t] = sic4dvar_dict['filtered_data']['reach_s'][0][t] * np.sum(sic4dvar_dict['filtered_data']['node_x'])
                else:
                    if len(sic4dvar_dict['filtered_data']['node_z']) < 2:
                        raise 'Need more than 2 sections/reaches to use Algo3.1 to calculate Qm!'
                    if len(sic4dvar_dict['filtered_data']['node_z']) >= 2 and len(sic4dvar_dict['filtered_data']['node_z']) <= 3:
                        SS1 = sic4dvar_dict['filtered_data']['node_z'][0][t]
                        SS2 = sic4dvar_dict['filtered_data']['node_z'][-1][t]
                    if len(sic4dvar_dict['filtered_data']['node_z']) > 3:
                        SS1 = (sic4dvar_dict['filtered_data']['node_z'][0][t] + 2.0 * sic4dvar_dict['filtered_data']['node_z'][1][t] + sic4dvar_dict['filtered_data']['node_z'][2][t]) / 4.0
                        index_x = np.where(abs(sic4dvar_dict['filtered_data']['node_x'][0] - sic4dvar_dict['filtered_data']['node_x'][0:]) < params.DX_Length)
                        index_x = index_x[0]
                        SS2 = (sic4dvar_dict['filtered_data']['node_z'][index_x[-3]][t] + 2.0 * sic4dvar_dict['filtered_data']['node_z'][index_x[-2]][t] + sic4dvar_dict['filtered_data']['node_z'][index_x[-1]][t]) / 4.0
                    SLOPEM1[t] = SS1 - SS2
        if sic4dvar_dict['param_dict']['run_type'] == 'seq':
            if not params.node_length:
                SLOPEM1 = sic4dvar_dict['filtered_data']['reach_s'] * (sic4dvar_dict['filtered_data']['node_x'][-1] - sic4dvar_dict['filtered_data']['node_x'][0])
            elif params.node_length:
                SLOPEM1 = sic4dvar_dict['filtered_data']['reach_s'] * (sic4dvar_dict['filtered_data']['node_x'][-1] - sic4dvar_dict['filtered_data']['node_x'][0])
                sic4dvar_dict['last_node_for_integral'] = len(sic4dvar_dict['filtered_data']['node_x'])
    else:
        SLOPEM1 = np.zeros(sic4dvar_dict['filtered_data']['node_z'][0].shape[0])
        for t in range(sic4dvar_dict['filtered_data']['node_z'][0].shape[0]):
            if len(sic4dvar_dict['filtered_data']['node_z']) < 2:
                logging.warning('WARNING: Need more than 2 sections/reaches to use Algo3.1 to calculate Qm!')
                sic4dvar_dict['output']['valid'] = 0
                sic4dvar_dict['last_node_for_integral'] = 0
                return (np.ones(sic4dvar_dict['filtered_data']['node_z'][0].shape[0]) * np.nan, sic4dvar_dict)
            if len(sic4dvar_dict['filtered_data']['node_z']) == 2:
                SS1 = sic4dvar_dict['filtered_data']['node_z'][0][t]
                SS2 = sic4dvar_dict['filtered_data']['node_z'][-1][t]
                sic4dvar_dict['last_node_for_integral'] = 2
            if len(sic4dvar_dict['filtered_data']['node_z']) >= 3:
                SS1 = sic4dvar_dict['filtered_data']['node_z'][0][t]
                SS2 = sic4dvar_dict['filtered_data']['node_z'][-1][t]
                '\n                # H.O. : Modification.\n                if not params.node_length:\n                    if abs(self.filtered_data[\'node_x\'][2]-self.filtered_data[\'node_x\'][0])>params.DX_Length: #modif D.Q\n                        params.DX_Length = abs(self.filtered_data[\'node_x\'][2]-self.filtered_data[\'node_x\'][0])\n                    index_x = np.where(abs(self.filtered_data[\'node_x\'][0]-self.filtered_data[\'node_x\'][0:])<=params.DX_Length)\n                    print("index_x=",index_x)\n                    print(abs(self.filtered_data[\'node_x\'][0]-self.filtered_data[\'node_x\'][0:]))\n                    print("DX_length=",params.DX_Length)\n                elif params.node_length:\n                    """\n                    temp=[]\n                    temp_sum=0\n                    for i in range(0,self.filtered_data[\'node_x\'].size):\n                        temp_sum = temp_sum + self.filtered_data[\'node_x\'][i]\n                        temp.append(temp_sum)\n                    """\n                    print(self.filtered_data[\'node_x\'][0]-self.filtered_data[\'node_x\'][0:])\n                    print(self.filtered_data[\'node_x\'][0]-temp[0:])\n                    index_x = np.where(abs(self.filtered_data[\'node_x\'][0]-temp[0:])<params.DX_Length)\n                    print("index_x=",index_x)\n                '
                sic4dvar_dict['last_node_for_integral'] = len(sic4dvar_dict['filtered_data']['node_z'])
            SLOPEM1[t] = SS1 - SS2
        if sic4dvar_dict['param_dict']['gnuplot_saving']:
            reach_id = verify_name_length(str(sic4dvar_dict['input_data']['reach_id']))
            output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', reach_id)
            if not Path(output_path).is_dir():
                Path(output_path).mkdir(parents=True, exist_ok=True)
            times2 = np.around(sic4dvar_dict['filtered_data']['reach_t'] / 3600 / 24)
            times2 = times2 - min(times2)
    return (SLOPEM1, sic4dvar_dict)

def compute_widths_from_breakpoints(height_breakpoints, poly_fits):
    """
    Given height breakpoints and corresponding polynomial fits (slope + intercept),
    return a list of (height, width) values at each breakpoint.
    
    Assumes:
    - height_breakpoints has N+1 values
    - poly_fits has N polynomials (1 per interval)
    - last height_breakpoint is included in last segment
    """
    height_breakpoints = np.array(height_breakpoints)
    widths = []
    for i in range(len(height_breakpoints)):
        h = height_breakpoints[i]
        if np.isnan(h):
            widths.append(np.nan)
            continue
        if i == len(height_breakpoints) - 1:
            p = poly_fits[-1]
        else:
            p = poly_fits[i]
        w = np.polyval(p, h)
        widths.append(w)
    return (widths, height_breakpoints)

def mike_method(param_dict, filtered_data, node_z, node_w, i, slope=[], algo='', input_data=[]):
    if param_dict['use_reach_slope']:
        ObsData = create_confluence_dict(filtered_data['reach_t'], node_z[i], node_w[i], filtered_data['reach_s'])
    elif algo != 'algo5':
        print('TEST0')
        print('len:', len(filtered_data['reach_t']), len(node_z[i]), len(node_w[i]), len(slope))
        ObsData = create_confluence_dict(filtered_data['reach_t'], node_z[i], node_w[i], slope)
    else:
        print('TEST1')
        print('len:', len(input_data['reach_t']), len(input_data['node_z'][i]), len(input_data['node_w'][i]), len(slope))
        ObsData = create_confluence_dict(input_data['reach_t'], input_data['node_z'][i], input_data['node_w'][i], slope)
    D = DomainHWS(ObsData)
    hws_obj = CalculateHWS(D, ObsData)
    if hasattr(hws_obj, 'area_fit'):
        results = [hws_obj.area_fit['w_break'], hws_obj.area_fit['h_break']]
        results2 = [hws_obj.wobs, hws_obj.hobs]
    if len(hws_obj.dAall) == 1:
        hws_obj.dAall = hws_obj.dAall[0]
    reach_dA = hws_obj.dAall
    h_breakpoints = hws_obj.h_breakpoints
    polyfits = hws_obj.poly_fits
    check_na_vec = np.vectorize(check_na)
    mask = ~check_na_vec(node_z[i]) & ~check_na_vec(node_w[i])
    new_node_z = node_z[i][mask]
    new_node_w = node_w[i][mask]
    if check_na(h_breakpoints[0]) or h_breakpoints[0] < 0.0:
        pass
        h_breakpoints[0] = np.nanmin(new_node_z)
    if check_na(h_breakpoints[3]) or h_breakpoints[3] < 0.0:
        pass
        h_breakpoints[3] = np.nanmax(new_node_z)
    results = compute_widths_from_breakpoints(np.array(h_breakpoints), np.array(polyfits))
    return (results, reach_dA)

def bathymetry_computation(node_w, node_z, param_dict, params, input_data=[], filtered_data=[], slope=[], force_method='', algo=''):
    node_xr = []
    node_yr = []
    dA = []
    cs_method = param_dict['cs_method']
    if force_method != '':
        cs_method = force_method
    logging.info(f'CS_METHOD: {cs_method}')
    for i in range(len(node_w)):
        if not params.pankaj_test:
            if cs_method == 'POM':
                results = f_approx_sections_v6(node_w[i], node_z[i], params.approx_section_params[0], params.approx_section_params[1], params.approx_section_params[2])
            elif cs_method == 'Igor':
                cs_i_w_low_bound0_array = np.ones(len(node_w[i])) * 10.0
                results = M(node_w[i], node_z[i], max_iter=params.LSMX, cor_z=None, inter_behavior=True, inter_behavior_min_thr=params.def_float_atol, inter_behavior_max_thr=params.DX_max_in, min_change_v_thr=0.0001, first_sweep='forward', cs_float_atol=params.def_float_atol, number_of_nodes=len(node_z), plot=False, cs_i_w_low_bound0_array=cs_i_w_low_bound0_array)
                results = f_approx_sections_v6(results[0], results[1], params.approx_section_params[0], params.approx_section_params[1], FSort=0)
                if params.quantile_bathy_experiment:
                    quantile_w_90 = np.quantile(node_w[i], 0.98)
                    if results[0][-2] < quantile_w_90:
                        results[0][-2] = quantile_w_90
                    if results[0][-1] < quantile_w_90:
                        results[0][-1] = quantile_w_90
            elif cs_method == 'Mike' and Confluence_HWS_method:
                results, _ = mike_method(param_dict, filtered_data, node_z, node_w, i, slope, algo, input_data)
            node_xr += [results[0]]
            node_yr += [results[1]]
            if param_dict['cs_plot_debug']:
                results_pom = f_approx_sections_v6(node_w[i], node_z[i], params.approx_section_params[0], params.approx_section_params[1], params.approx_section_params[2])
                results_igor = M(node_w[i], node_z[i], max_iter=params.LSMX, cor_z=None, inter_behavior=True, inter_behavior_min_thr=params.def_float_atol, inter_behavior_max_thr=params.DX_max_in, min_change_v_thr=0.0001, first_sweep='forward', cs_float_atol=params.def_float_atol, number_of_nodes=len(node_z), plot=False)
                results_igor = f_approx_sections_v6(results_igor[0], results_igor[1], params.approx_section_params[0], params.approx_section_params[1], FSort=0)
                if params.quantile_bathy_experiment:
                    if results_igor[0][-2] < quantile_w_90:
                        results_igor[0][-2] = quantile_w_90
                    if results_igor[0][-1] < quantile_w_90:
                        results_igor[0][-1] = quantile_w_90
                if Confluence_HWS_method:
                    results_mike, dA_mike_2 = mike_method(param_dict, filtered_data, node_z, node_w, i, slope)
                    plt.plot(results_mike[0], results_mike[1], label='Mike')
                plt.plot(results_pom[0], results_pom[1], label='POM')
                print(results_pom[0].shape, results_pom[1].shape)
                plt.plot(results_igor[0], results_igor[1], label='Igor')
                print(results_igor[0].shape, results_igor[1].shape)
                plt.plot(node_w[i], node_z[i], marker='.', linestyle='None', label='orig pts')
                plt.legend(loc='upper right')
                plt.show()
                plt.savefig('plot_debug.png')
                plt.clf()
    if param_dict['gnuplot_saving']:
        node_x = input_data['node_x']
        times2 = np.around(input_data['reach_t'] / 3600 / 24)
        times2 = times2 - np.nanmin(times2)
        nodes2 = (node_x - node_x[0]) / 1000
        reach_id = str(input_data['reach_id'])
        reach_id = verify_name_length(reach_id)
        output_path = param_dict['output_dir'].joinpath('gnuplot_data', str(reach_id))
        if not Path(output_path).is_dir():
            Path(output_path).mkdir(parents=True, exist_ok=True)
        output_path = param_dict['output_dir'].joinpath('gnuplot_data', str(reach_id), 'cs')
    return (node_xr, node_yr, dA)

def slope_modification(wse, slope):
    wse_mean = []
    for t in range(wse.shape[1]):
        wse_mean.append(np.nanmean(wse[:, t]))
    wse_mean = np.array(wse_mean)
    increasing_index = np.argsort(wse_mean)
    new_wse_mean = deepcopy(wse_mean[increasing_index])
    slope = deepcopy(slope[increasing_index])
    return (new_wse_mean, slope, increasing_index, wse_mean)

def compute_slope(sic4dvar_dict, params):
    SLOPEM1, sic4dvar_dict = slope_computation(sic4dvar_dict)
    SLOPEM1_orig = deepcopy(SLOPEM1)
    sic4dvar_dict['output']['SLOPEM1_orig'] = SLOPEM1_orig
    length_reach = np.abs(sic4dvar_dict['filtered_data']['node_x'][-1] - sic4dvar_dict['filtered_data']['node_x'][0])
    if not params.static_slope:
        if np.mean(SLOPEM1) / length_reach < 0.0002:
            use_constant_slope = False
            use_smoothed_slope = True
            logging.info(f'Mean slope is low <2e-4: {np.mean(SLOPEM1) / length_reach}. Using smoothing.')
        else:
            use_constant_slope = True
            use_smoothed_slope = False
            logging.info(f'Mean slope is >=2e-4: {np.mean(SLOPEM1) / length_reach}. Using constant slope.')
    else:
        use_constant_slope = True
        use_smoothed_slope = False
    if params.slope_smooth_wse_ranking:
        array_to_use, SLOPEM1_reordered, increasing_index, wse_mean = slope_modification(sic4dvar_dict['filtered_data']['node_z'], SLOPEM1)
        correlation = (array_to_use[-1] - array_to_use[0]) / array_to_use.shape[0]
        behavior = 'increase'
        inter_behavior = True
    else:
        correlation = sic4dvar_dict['cort_slope']
        array_to_use = sic4dvar_dict['filtered_data']['reach_t']
        SLOPEM1_reordered = SLOPEM1
        behavior = ''
        inter_behavior = False
    SLOPEM1_2D = []
    for n in range(0, len(sic4dvar_dict['filtered_data']['node_z'])):
        SLOPEM1_2D.append(SLOPEM1_reordered)
    SLOPEM1_2D = np.array(SLOPEM1_2D)
    SLOPEM1_2D = K(dim=0, value0_array=SLOPEM1_2D, base0_array=array_to_use, max_iter=params.slope_smooth_max_iter, cor=correlation, always_run_first_iter=False, behavior=behavior, inter_behavior=inter_behavior, inter_behavior_min_thr=params.def_float_atol, inter_behavior_max_thr=params.DX_max_in, check_behavior='', min_change_v_thr=0.0001, plot=False, plot_title='Relaxation sweep in time SLOPEM1 without Interchange', clean_run=True, debug_mode=False)
    tmp = deepcopy(SLOPEM1)
    SLOPEM1_smoothed = SLOPEM1_2D[0]
    if params.slope_smooth_wse_ranking:
        tmp_slope = []
        for t in range(0, len(SLOPEM1_smoothed)):
            index = np.where(increasing_index == t)[0][0]
            tmp_slope.append(SLOPEM1_smoothed[index])
        SLOPEM1_smoothed = np.array(tmp_slope)
    if sic4dvar_dict['param_dict']['gnuplot_saving']:
        reach_id = verify_name_length(str(sic4dvar_dict['input_data']['reach_id']))
        output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', reach_id)
        if not Path(output_path).is_dir():
            Path(output_path).mkdir(parents=True, exist_ok=True)
        times2 = np.around(sic4dvar_dict['filtered_data']['reach_t'] / 3600 / 24)
        times2 = times2 - min(times2)
        gnuplot_save_slope(SLOPEM1_orig, times2, output_path.joinpath('slope_orig'))
        gnuplot_save_slope(SLOPEM1_smoothed, times2, output_path.joinpath('slope_smooth'))
    if use_smoothed_slope:
        sic4dvar_dict['output']['SLOPEM1'] = SLOPEM1_smoothed
    sic4dvar_dict['output']['SLOPEM1_smoothed'] = deepcopy(SLOPEM1_smoothed)
    if use_constant_slope:
        SLOPEM1_mean = 0.0
        time_scaling = 0.0
        for t in range(1, len(SLOPEM1)):
            SLOPEM1_mean += (SLOPEM1[t] + SLOPEM1[t - 1]) / 2 * (sic4dvar_dict['filtered_data']['reach_t'][t] - sic4dvar_dict['filtered_data']['reach_t'][t - 1])
            time_scaling += sic4dvar_dict['filtered_data']['reach_t'][t] - sic4dvar_dict['filtered_data']['reach_t'][t - 1]
        for t in range(0, len(SLOPEM1)):
            SLOPEM1[t] = SLOPEM1_mean / time_scaling
        if sic4dvar_dict['param_dict']['gnuplot_saving']:
            reach_id = verify_name_length(str(sic4dvar_dict['input_data']['reach_id']))
            output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', reach_id)
            if not Path(output_path).is_dir():
                Path(output_path).mkdir(parents=True, exist_ok=True)
            times2 = np.around(sic4dvar_dict['filtered_data']['reach_t'] / 3600 / 24)
            times2 = times2 - min(times2)
            gnuplot_save_slope(SLOPEM1, times2, output_path.joinpath('slope_constant'))
        sic4dvar_dict['output']['SLOPEM1_constant'] = SLOPEM1
    if not sic4dvar_dict['output']['valid']:
        sic4dvar_dict['output']['valid'] = 0
        logging.warning('Slope not valid.')
        return (SLOPEM1, sic4dvar_dict)
    return (SLOPEM1, sic4dvar_dict)

def compute_bathymetry(sic4dvar_dict, params, SLOPEM1):
    sic4dvar_dict['input_data']['node_xr'], sic4dvar_dict['input_data']['node_yr'], _ = bathymetry_computation(node_w=sic4dvar_dict['filtered_data']['node_w'], node_z=sic4dvar_dict['filtered_data']['node_z'], param_dict=sic4dvar_dict['param_dict'], params=params, input_data=sic4dvar_dict['input_data'], filtered_data=sic4dvar_dict['filtered_data'], slope=SLOPEM1)
    count_w = 0
    count_z = 0
    for t in range(0, len(sic4dvar_dict['input_data']['reach_w'])):
        if check_na(sic4dvar_dict['input_data']['reach_w'][t]):
            count_w += 1
        if check_na(sic4dvar_dict['input_data']['reach_z'][t]):
            count_z += 1
    if count_w >= len(sic4dvar_dict['input_data']['reach_w']) - 1 or count_z >= len(sic4dvar_dict['input_data']['reach_z']) - 1:
        logging.warning("All values in reach_w or reach_z are NA. Can't compute reach bathymetry.")
        sic4dvar_dict['input_data']['reach_xr'] = np.array(np.ones(10) * np.nan)
        sic4dvar_dict['input_data']['reach_yr'] = np.array(np.ones(10) * np.nan)
    else:
        logging.info('Computing reach bathymetry.')
        sic4dvar_dict['input_data']['reach_xr'], sic4dvar_dict['input_data']['reach_yr'], _ = bathymetry_computation(node_w=[sic4dvar_dict['input_data']['reach_w']], node_z=[sic4dvar_dict['input_data']['reach_z']], param_dict=sic4dvar_dict['param_dict'], params=params, input_data=sic4dvar_dict['input_data'], filtered_data=sic4dvar_dict['filtered_data'], slope=SLOPEM1)
        sic4dvar_dict['input_data']['reach_xr'] = sic4dvar_dict['input_data']['reach_xr'][0]
        sic4dvar_dict['input_data']['reach_yr'] = sic4dvar_dict['input_data']['reach_yr'][0]
    node_x = sic4dvar_dict['input_data']['node_x']
    reach_id = str(sic4dvar_dict['input_data']['reach_id'])
    reach_id = verify_name_length(reach_id)
    times2 = np.around(sic4dvar_dict['input_data']['reach_t'] / 3600 / 24)
    times2 = times2 - np.nanmin(times2)
    nodes2 = (node_x - node_x[0]) / 1000
    tmp_min = []
    for i in range(0, len(sic4dvar_dict['input_data']['node_yr'])):
        tmp_min.append(np.nanmin(sic4dvar_dict['input_data']['node_yr'][i]))
    if sic4dvar_dict['param_dict']['gnuplot_saving']:
        output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', reach_id)
        if not Path(output_path).is_dir():
            Path(output_path).mkdir(parents=True, exist_ok=True)
        gnuplot_save(nodes2, times2, sic4dvar_dict['input_data']['node_z_ini'], sic4dvar_dict['input_data']['node_w_ini'], output_path.joinpath('out_wse_w' + '_ini'), tmp_min, 2)
        if sic4dvar_dict['param_dict']['run_type'] == 'seq':
            gnuplot_save(nodes2, times2, sic4dvar_dict['tmp_interp_values'][0], sic4dvar_dict['input_data']['node_w_ini'], output_path.joinpath('out_wse' + '_deviation'), tmp_min, 2)
            gnuplot_save(nodes2, times2, sic4dvar_dict['tmp_interp_values'][1], sic4dvar_dict['tmp_interp_values_w'][0], output_path.joinpath('out_wse_w' + '_relax_space1'), tmp_min, 2)
            gnuplot_save(nodes2, times2, sic4dvar_dict['tmp_interp_values'][2], sic4dvar_dict['tmp_interp_values_w'][1], output_path.joinpath('out_wse_w' + '_interpolated'), tmp_min, 2)
            gnuplot_save(nodes2, times2, sic4dvar_dict['tmp_interp_values'][3], sic4dvar_dict['tmp_interp_values_w'][2], output_path.joinpath('out_wse_w' + '_relax_space2'), tmp_min, 2)
            gnuplot_save(nodes2, times2, sic4dvar_dict['tmp_interp_values'][4], sic4dvar_dict['tmp_interp_values_w'][3], output_path.joinpath('out_wse_w' + '_final'), tmp_min, 1)
            if len(sic4dvar_dict['tmp_interp_values']) > 5:
                gnuplot_save(nodes2, times2, sic4dvar_dict['tmp_interp_values'][5], sic4dvar_dict['input_data']['node_w_ini'], output_path.joinpath('out_wse' + '_deviation_new'), tmp_min, 2)
        else:
            gnuplot_save(nodes2, times2, sic4dvar_dict['tmp_interp_values'][0], sic4dvar_dict['tmp_interp_values_w'][0], output_path.joinpath('out_wse_w' + '_relax_space1'), tmp_min, 2)
            gnuplot_save(nodes2, times2, sic4dvar_dict['tmp_interp_values'][1], sic4dvar_dict['tmp_interp_values_w'][1], output_path.joinpath('out_wse_w' + '_interpolated'), tmp_min, 2)
            gnuplot_save(nodes2, times2, sic4dvar_dict['tmp_interp_values'][2], sic4dvar_dict['tmp_interp_values_w'][2], output_path.joinpath('out_wse_w' + '_relax_space2'), tmp_min, 2)
            gnuplot_save(nodes2, times2, sic4dvar_dict['tmp_interp_values'][3], sic4dvar_dict['tmp_interp_values_w'][3], output_path.joinpath('out_wse_w' + '_final'), tmp_min, 1)
            if len(sic4dvar_dict['tmp_interp_values']) > 4:
                gnuplot_save(nodes2, times2, sic4dvar_dict['tmp_interp_values'][5], sic4dvar_dict['input_data']['node_w_ini'], output_path.joinpath('out_wse' + '_deviation_new'), tmp_min, 2)
        gnuplot_save_list(nodes2, times2, sic4dvar_dict['input_data']['node_yr'], sic4dvar_dict['input_data']['node_xr'], output_path.joinpath('out_z_w' + '_bathy'), tmp_min, 2)
    if False:
        for n in range(0, len(sic4dvar_dict['input_data']['node_xr'])):
            for i in range(0, len(sic4dvar_dict['input_data']['node_xr'][n])):
                sic4dvar_dict['input_data']['node_xr'][n][i] = sic4dvar_dict['input_data']['node_xr'][n][i] * 2
    sic4dvar_dict['input_data']['node_a'], sic4dvar_dict['input_data']['node_p'], sic4dvar_dict['input_data']['node_r'], sic4dvar_dict['input_data']['node_w_simp'], sic4dvar_dict['bb'] = call_func_APR(sic4dvar_dict['filtered_data']['node_w'], sic4dvar_dict['filtered_data']['node_z'], sic4dvar_dict['input_data']['node_xr'], sic4dvar_dict['input_data']['node_yr'], params, sic4dvar_dict['param_dict'])
    tmp40 = []
    tmp41 = []
    tmp42 = []
    tmp43 = []
    tmp44 = []
    for n in range(0, sic4dvar_dict['filtered_data']['node_z'].shape[0]):
        tmp41.append(sic4dvar_dict['filtered_data']['node_z'][n, sic4dvar_dict['filtered_data']['node_z'][n, :].argmax()])
        tmp43.append(sic4dvar_dict['filtered_data']['node_w'][n, sic4dvar_dict['filtered_data']['node_w'][n, :].argmax()])
        tmp40.append(sic4dvar_dict['input_data']['node_xr'][n][sic4dvar_dict['input_data']['node_xr'][n].argmin()])
        tmp44.append(sic4dvar_dict['input_data']['node_xr'][n].argmin())
    apr_array = {'node_w_simp': sic4dvar_dict['input_data']['node_w_simp'], 'node_a': sic4dvar_dict['input_data']['node_a'], 'node_p': sic4dvar_dict['input_data']['node_p'], 'node_r': sic4dvar_dict['input_data']['node_r']}
    bathymetry_array = {'node_xr': sic4dvar_dict['input_data']['node_xr'], 'node_yr': sic4dvar_dict['input_data']['node_yr'], 'reach_xr': sic4dvar_dict['input_data']['reach_xr'], 'reach_yr': sic4dvar_dict['input_data']['reach_yr']}
    return (sic4dvar_dict, bathymetry_array, apr_array)

def compute_relative_elevation(elev_2D):
    """
    Compute relative elevation for each node's bathymetry profile.
    
    Parameters
    ----------
    elev_2D : np.ndarray
        2D array of elevation values (n_nodes, n_pts)

    """
    if len(elev_2D.shape) > 1:
        node_elev_min = np.nanmin(elev_2D, axis=1, keepdims=True)
    else:
        node_elev_min = np.nanmin(elev_2D, keepdims=True)
    elev_rel_2D = elev_2D - node_elev_min
    return (elev_rel_2D, node_elev_min)

def compute_absolute_elevation(elev_rel, node_elev_min):
    """
    Docstring pour compute_absolute_elevation
    
    :param elev_rel_2D: Description
    :param node_elev_min: Description
    """
    elev_abs = elev_rel + node_elev_min
    return elev_abs

def shift_bathy_profile_to_fit_global_mean(elev_rel_2D, width_2D):
    """
    Shift each node's bathymetry profile so its mean elevation matches the global mean.
    
    Parameters
    ----------
    elev_2D : np.ndarray
        2D array of elevation values (n_nodes, n_pts)
    width_2D : np.ndarray
        2D array of width values (n_nodes, n_pts)
    
    Returns
    -------
    elev_norm : np.ndarray
        Normalized elevation values (n_nodes, n_pts)
    width_2D : np.ndarray
        Width values unchanged (n_nodes, n_pts)
    """
    n_nodes = elev_rel_2D.shape[0]
    global_mean = np.nanmean(elev_rel_2D)
    elev_rel_norm = np.zeros_like(elev_rel_2D)
    for n in range(n_nodes):
        node_mean = np.nanmean(elev_rel_2D[n, :])
        elev_rel_norm[n, :] = elev_rel_2D[n, :] - (node_mean - global_mean)
    return (elev_rel_norm, width_2D)

def compute_sigma_bounds(elev):
    elev_norm_without_outliers = elev[(elev > np.nanquantile(elev, 0.05)) & (elev < np.nanquantile(elev, 0.95))]
    elev_norm_without_outliers_mean = np.nanmean(elev_norm_without_outliers)
    elev_std_positive = np.nanstd(elev_norm_without_outliers[elev_norm_without_outliers >= elev_norm_without_outliers_mean])
    elev_std_negative = np.nanstd(elev_norm_without_outliers[elev_norm_without_outliers < elev_norm_without_outliers_mean])
    elev_rel_min = elev_norm_without_outliers_mean - 2 * elev_std_negative
    elev_rel_max = elev_norm_without_outliers_mean + 2 * elev_std_positive
    return (elev_rel_min, elev_rel_max)

def aggregate_node_bathy_to_reach(elev_2D, width_2D, option='2sigma', nb_points=10):
    """
    Aggregate node-level bathymetry to reach-level by interpolating and averaging.
    
    Parameters
    ----------
    elev_2D : np.ndarray
        2D array of elevation values (n_nodes, n_pts)
    width_2D : np.ndarray
        2D array of width values (n_nodes, n_pts)
    option : str
        'all' - use full elevation range (min to max across all nodes)
        'common' - use common elevation range (max of mins to min of maxs)
    nb_points : int
        Number of elevation points for interpolation
    
    Returns
    -------
    reach_width : np.ndarray
        Averaged width values at each elevation (nb_points,)
    reach_elev : np.ndarray
        Elevation values (nb_points,)
    """
    n_nodes = elev_2D.shape[0]
    reach_elev_abs = np.nanmean(elev_2D)
    elev_rel_2D, node_elev_min = compute_relative_elevation(elev_2D)
    elev_rel_norm_2D, _ = shift_bathy_profile_to_fit_global_mean(elev_rel_2D, width_2D)
    if option == 'common':
        elev_rel_min = np.nanmax(np.nanmin(elev_rel_norm_2D, axis=1))
        elev_rel_max = np.nanmin(np.nanmax(elev_rel_norm_2D, axis=1))
    elif option == '2sigma':
        elev_rel_min, elev_rel_max = compute_sigma_bounds(elev_rel_norm_2D)
    elif option == '80%common':
        elev_rel_mins = np.nanmin(elev_rel_norm_2D, axis=1)
        elev_rel_maxs = np.nanmax(elev_rel_norm_2D, axis=1)
        elev_rel_min = np.nanpercentile(elev_rel_mins, 80)
        elev_rel_max = np.nanpercentile(elev_rel_maxs, 20)
    elif option == '50%common':
        elev_rel_mins = np.nanmin(elev_rel_norm_2D, axis=1)
        elev_rel_maxs = np.nanmax(elev_rel_norm_2D, axis=1)
        elev_rel_min = np.nanpercentile(elev_rel_mins, 50)
        elev_rel_max = np.nanpercentile(elev_rel_maxs, 50)
    elif option == '20%common':
        elev_rel_mins = np.nanmin(elev_rel_norm_2D, axis=1)
        elev_rel_maxs = np.nanmax(elev_rel_norm_2D, axis=1)
        elev_rel_min = np.nanpercentile(elev_rel_mins, 20)
        elev_rel_max = np.nanpercentile(elev_rel_maxs, 80)
    elif option == 'all':
        elev_rel_min = np.nanmin(elev_rel_norm_2D)
        elev_rel_max = np.nanmax(elev_rel_norm_2D)
    else:
        raise ValueError(f"Unknown option '{option}' for aggregate_node_bathy_to_reach")
    logging.debug(f'Aggregating bathymetry using elevation range: {elev_rel_min} to {elev_rel_max} (option: {option})')
    reach_rel_elev = np.linspace(elev_rel_min, elev_rel_max, nb_points)
    width_interp = np.zeros((n_nodes, nb_points))
    for n in range(n_nodes):
        elev = elev_rel_norm_2D[n, :]
        width = width_2D[n, :]
        valid = ~np.isnan(elev) & ~np.isnan(width)
        if np.sum(valid) < 2:
            width_interp[n, :] = np.nan
            continue
        elev_valid = elev[valid]
        width_valid = width[valid]
        sort_idx = np.argsort(elev_valid)
        elev_sorted = elev_valid[sort_idx]
        width_sorted = width_valid[sort_idx]
        width_interp[n, :] = np.interp(reach_rel_elev, elev_sorted, width_sorted)
    reach_width = np.nanmean(width_interp, axis=0)
    reach_abs_elev = reach_elev_abs + reach_rel_elev - np.nanmean(reach_rel_elev)
    return (reach_width, reach_rel_elev, reach_abs_elev)

def compute_z_bed(node_w_simp, node_z, node_xr, node_yr, Zb_acc):
    Wmin = 10000
    Wmean = []
    for i in range(len(node_w_simp)):
        Wmean.append(min(node_w_simp[i]))
    Wmean = np.average(Wmean)
    z_bed = np.ones(node_z.shape[0]) * np.nan
    Zb_acc_val = Zb_acc[0] if np.ndim(Zb_acc) > 0 else Zb_acc
    for n in range(0, len(node_z)):
        Wmin = np.nanmin(node_xr[n])
        z_bed[n] = node_yr[n][0] + params.algo_bounds[1][1] + Zb_acc_val * (Wmean / Wmin)
    return z_bed

def compute_wet_area(bathy_width, bathy_elev, Zb):
    A0 = np.nanmin(bathy_width, axis=1) * (np.nanmin(bathy_elev, axis=1) - Zb)
    return A0