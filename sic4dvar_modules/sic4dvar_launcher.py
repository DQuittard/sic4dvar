import logging
from copy import deepcopy
from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import brent, minimize_scalar
from pathlib import Path
import datetime
import sic4dvar_params as params
from lib.lib_dates import daynum_to_date
from lib.lib_verif import verify_name_length, check_na
from lib.lib_indicators import integrated_mean, integrated_variance
from sic4dvar_algos.X653 import Z3, Z1
from sic4dvar_algos.algo5 import algo5
from sic4dvar_functions.sic4dvar_gnuplot_save import gnuplot_save_cs, gnuplot_save_q, gnuplot_save_var, gnuplot_save_cs, gnuplot_save_var
from sic4dvar_functions.sic4dvar_helper_functions import build_output_q_masked_array, build_output_2d_masked_array, compute_mean_var_from_2D_array, get_external_friction_data, grad_variance, interp_pdf_tables, compute_mean_elevation_profile, make_mask_period
from sic4dvar_modules.sic4dvar_compute_slope_and_bathymetry import call_func_APR, compute_bathymetry, compute_slope, compute_z_bed
from sic4dvar_modules.sic4dvar_create_filtered_arrays import create_filtered_arrays, fill_nan_with_neighbours
from sic4dvar_modules.sic4dvar_filtering import filter_based_on_config, remove_unuseable_nodes
from sic4dvar_modules.sic4dvar_launch_algo5 import launch_algo5
from sic4dvar_modules.sic4dvar_qwbm_replace import replace_prior

def set_correlation_parameters(node_x, reach_t):
    other_sic_params = {}
    if params.corx_option == 0:
        corx = 200.0
    elif params.corx_option == 1:
        corx = np.mean(np.diff(node_x))
    elif params.corx_option == 2:
        corx = 200.0
    corx_array = np.ones(len(node_x)) * corx
    cort = params.cort
    cort_width = cort
    cort_tmp = []
    cort_slope = 0.0
    for t in range(0, len(reach_t)):
        if not check_na(reach_t[t]):
            cort_tmp.append(reach_t[t])
    cort_slope = np.mean(np.diff(cort_tmp))
    if params.override_cort:
        cort = cort_slope
    cort_wse = cort
    other_sic_params['cort_wse'] = cort_wse
    other_sic_params['cort_slope'] = 0.0
    return (corx, corx_array, cort_wse, cort_width, other_sic_params)

def create_monthly_mean_series(swot_times_in_seconds, q_monthly_mean):
    ref = np.datetime64('2000-01-01')
    swot_datetimes = ref + swot_times_in_seconds * np.timedelta64(1, 's')
    pred_times_array = np.ma.asarray(swot_datetimes)
    if np.ma.isMaskedArray(pred_times_array):
        pred_times_values = pred_times_array.compressed()
    else:
        pred_times_values = np.asarray(pred_times_array).ravel()
    pred_time_series = pd.to_datetime(pd.Series(pred_times_values), errors='coerce').to_frame(name='swot_times')
    pred_time_series['month'] = pred_time_series['swot_times'].dt.month
    pred_time_series['day'] = pred_time_series['swot_times'].dt.day
    pred_time_series['swot_days'] = swot_times_in_seconds / 86400
    q_monthly = []
    year = 4
    for index, row in pred_time_series.iterrows():
        if pd.isnull(row['day']) or pd.isnull(row['month']):
            q3 = np.nan
            q_monthly.append(q3)
            continue
        current_month = int(row['month'])
        current_day = int(row['day'])
        if current_day >= 15:
            q1 = q_monthly_mean[current_month - 1]
            if current_month == 12:
                q2 = q_monthly_mean[0]
            else:
                q2 = q_monthly_mean[current_month - 1 + 1]
            t1 = datetime.datetime(year, current_month, 15)
            if current_month == 12:
                t2 = datetime.datetime(year + 1, 1, 15)
            else:
                t2 = datetime.datetime(year, current_month + 1, 15)
        else:
            if current_month == 1:
                q1 = q_monthly_mean[11]
            else:
                q1 = q_monthly_mean[current_month - 1 - 1]
            q2 = q_monthly_mean[current_month - 1]
            if current_month == 1:
                t1 = datetime.datetime(year - 1, 12, 15)
            else:
                t1 = datetime.datetime(year, current_month - 1, 15)
            t2 = datetime.datetime(year, current_month, 15)
        t3 = datetime.datetime(year, current_month, current_day)
        epoch_0001 = datetime.datetime(year, 1, 1)
        t1_seconds = (t1 - epoch_0001).total_seconds() / 86400
        t2_seconds = (t2 - epoch_0001).total_seconds() / 86400
        t3_seconds = (t3 - epoch_0001).total_seconds() / 86400
        from sic4dvar_functions.sic4dvar_helper_functions import interp_pdf_tables
        q3 = interp_pdf_tables(n=1, u=t3_seconds, x=[t1_seconds, t2_seconds], y=[q1, q2])
        q_monthly.append(q3)
    q_monthly_series = q_monthly
    return q_monthly_series

def plot_time(swot_times_in_seconds, q_monthly, Qt=None, Qt_smoothed=None, source_q_col=None, source_t_col=None, reach_id=None, Q_2=None):
    from matplotlib import pyplot as plt
    ref = np.datetime64('2000-01-01')
    swot_datetimes = ref + swot_times_in_seconds * np.timedelta64(1, 's')
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(swot_datetimes, q_monthly, marker='o', label='q_monthly (interpolated)')
    if source_q_col is not None and source_t_col is not None:
        ax.plot(source_t_col, source_q_col, marker='+', label='station q')
    if Qt is not None:
        ax.plot(swot_datetimes, Qt, marker='x', label='Qt')
    if Qt_smoothed is not None:
        ax.plot(swot_datetimes, Qt_smoothed, marker='s', label='Qt_smoothed')
    if Q_2 is not None:
        ax.plot(swot_datetimes, Q_2, marker='d', label='Q_2')
    ax.set_xlabel('Date')
    ax.set_ylabel('Discharge (m³/s)')
    ax.set_title(f'q_monthly vs station_q — reach {reach_id}')
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

def sic4dvar_preprocessing(sic4dvar_dict, params, reach_number=0):
    sic4dvar_dict['data_is_useable'] = True
    sic4dvar_dict['input_data']['create_series_from_monthly_q'] = False
    if params.create_series_from_monthly_q:
        sic4dvar_dict['input_data']['create_series_from_monthly_q'] = True
        logging.info('Creating time series from monthly Q data.')
        if False:
            for i in range(0, sic4dvar_dict['input_data']['q_monthly_mean'].size):
                sic4dvar_dict['input_data']['q_monthly_mean'][i] = np.nan
        nb_nan = 0.0
        for i in range(0, 12):
            if check_na(sic4dvar_dict['input_data']['q_monthly_mean'][i]):
                nb_nan += 1
        if nb_nan > 0.0:
            logging.warning("Can't create time series from monthly Q data.")
            sic4dvar_dict['input_data']['create_series_from_monthly_q'] = False
            logging.info('Fallback to regular model prior.')
        else:
            logging.info('Monthly Q data is valid. Creating time series from monthly Q data.')
            sic4dvar_dict['input_data']['q_monthly_series'] = create_monthly_mean_series(swot_times_in_seconds=sic4dvar_dict['input_data']['reach_t'], q_monthly_mean=sic4dvar_dict['input_data']['q_monthly_mean'])
            logging.info('Finished creating time series from monthly Q data.')
    logging.info('Runs launch_sic4dvar over %d reaches' % reach_number)
    if sic4dvar_dict['param_dict']['run_type'] == 'set':
        for i in range(0, reach_number):
            sic4dvar_dict['output']['valid_a5_sets'].append(1)
    sic4dvar_dict = filter_based_on_config(sic4dvar_dict)
    logging.info('Finished filtering the data.')
    if len(sic4dvar_dict['input_data']['node_x']) == 0:
        sic4dvar_dict['data_is_useable'] = False
        logging.warning('Empty SWORD node data.')
    if len(sic4dvar_dict['input_data']['node_x']) != len(sic4dvar_dict['input_data']['node_z']):
        sic4dvar_dict['data_is_useable'] = False
        logging.warning('Number of nodes is different between SWORD and SWOT file !')
        logging.warning('node_x length: %d, node_z length: %d' % (len(sic4dvar_dict['input_data']['node_x']), len(sic4dvar_dict['input_data']['node_z'])))
    logging.info('Finished checking nodes not empty + same size as in SWORD.')
    if params.extrapolation and sic4dvar_dict['data_is_useable']:
        logging.info('Extrapolating wse data')
        if True:
            from sic4dvar_modules.sic4dvar_extrapolation_new import Extrapolation
            corx, corx_array, cort_wse, cort_width, other_sic_params = set_correlation_parameters(sic4dvar_dict['input_data']['node_x'], sic4dvar_dict['input_data']['reach_t'])
            for key in other_sic_params:
                sic4dvar_dict[key] = other_sic_params[key]
            sic4dvar_dict['input_data']['node_z_ini'] = deepcopy(sic4dvar_dict['input_data']['node_z'])
            sic4dvar_dict['input_data']['node_w_ini'] = deepcopy(sic4dvar_dict['input_data']['node_w'])
            sic4dvar_dict['input_data']['node_t_ini'] = deepcopy(sic4dvar_dict['input_data']['node_t'])
            if params.exclude_invalid_overpasses:
                logging.info('Excluding invalid overpasses for extrapolation.')
                valid_overpasses = []
                for t in range(0, len(sic4dvar_dict['input_data']['reach_t'])):
                    for n in range(0, len(sic4dvar_dict['input_data']['node_x'])):
                        if not check_na(sic4dvar_dict['input_data']['node_z'][n, t]):
                            valid_overpasses.append(t)
                            break
                valid_overpasses = np.array(valid_overpasses)
                if valid_overpasses.size > 0:
                    logging.info(f'Valid overpasses: {valid_overpasses}')
                else:
                    logging.error('No valid overpass. Stopping.')
                    sic4dvar_dict['data_is_useable'] = False
                if sic4dvar_dict['data_is_useable']:
                    mean_elevation_profile = compute_mean_elevation_profile(sic4dvar_dict['input_data']['node_z'][:, valid_overpasses], sic4dvar_dict['input_data']['reach_t'][valid_overpasses])
                    mean_width_profile = compute_mean_elevation_profile(sic4dvar_dict['input_data']['node_w'][:, valid_overpasses], sic4dvar_dict['input_data']['reach_t'][valid_overpasses])
                    node_z_valid_t = deepcopy(sic4dvar_dict['input_data']['node_z'][:, valid_overpasses])
                    node_w_valid_t = deepcopy(sic4dvar_dict['input_data']['node_w'][:, valid_overpasses])
                valid_sections = []
                for n in range(0, len(sic4dvar_dict['input_data']['node_x'])):
                    if not check_na(mean_elevation_profile[n]):
                        valid_sections.append(n)
                        continue
                valid_sections = np.array(valid_sections)
                if valid_sections.size > 0:
                    logging.info(f'Valid sections: {valid_sections}')
                else:
                    logging.error('No valid section. Stopping.')
                    sic4dvar_dict['data_is_useable'] = False
                if sic4dvar_dict['data_is_useable']:
                    filtered_mean_elevation_profile = mean_elevation_profile[valid_sections]
                    filtered_mean_width_profile = mean_width_profile[valid_sections]
                    filtered_mean_elevation_profile = filtered_mean_elevation_profile.reshape(filtered_mean_elevation_profile.shape[0], -1)
                    filtered_mean_width_profile = filtered_mean_width_profile.reshape(filtered_mean_width_profile.shape[0], -1)
                    from sic4dvar_modules.sic4dvar_extrapolation_new import igor_method
                    filtered_mean_elevation_profile = igor_method(filtered_mean_elevation_profile, corx, sic4dvar_dict['input_data']['node_x'][valid_sections], LSMX=10, eps1=0.0001, eps2=0.0001)
                    for n in range(0, len(filtered_mean_width_profile)):
                        if check_na(filtered_mean_width_profile[n]):
                            filtered_mean_width_profile[n] = np.nanmean(filtered_mean_width_profile)
                    node_z, node_w, optionnal_output_dict = Extrapolation(node_z=node_z_valid_t[valid_sections, :], node_w=node_w_valid_t[valid_sections, :], node_x=sic4dvar_dict['input_data']['node_x'][valid_sections], reach_t=sic4dvar_dict['input_data']['reach_t'][valid_overpasses], corx=corx, corx_array=corx_array, cort_wse=cort_wse, cort_width=cort_width, gnuplot_saving=sic4dvar_dict['param_dict']['gnuplot_saving'], reach_id=sic4dvar_dict['input_data']['reach_id'], output_dir=sic4dvar_dict['param_dict']['output_dir'], run_type=sic4dvar_dict['param_dict']['run_type'], use_large_deviations=params.large_deviations, start_from_downstream=params.start_from_downstream, run_preprocessing=params.run_preprocessing, run_extrapolation=params.run_extrapolation, pooling=params.pooling, mean_elevation_profile=filtered_mean_elevation_profile, mean_width_profile=filtered_mean_width_profile)
                    print('NODE_Z SECTION:', node_z[:, 0])
                    print(node_z.shape)
                    sic4dvar_dict['observed_nodes'] = valid_sections
                    sic4dvar_dict['list_to_keep'] = valid_overpasses
                    valid_set = set(valid_overpasses)
                    invalid_overpasses = [i for i in range(sic4dvar_dict['input_data']['node_z_ini'].shape[1]) if i not in valid_set]
                    sic4dvar_dict['removed_indices'] = np.array(invalid_overpasses)
                    print('Valid sections:', sic4dvar_dict['observed_nodes'])
                    print('Valid overpasses:', sic4dvar_dict['list_to_keep'])
                    print('Removed indices:', sic4dvar_dict['removed_indices'])
                    sic4dvar_dict['input_data']['tmp_filtered_node_z'] = deepcopy(node_z)
                    sic4dvar_dict['input_data']['tmp_filtered_node_w'] = deepcopy(node_w)
                    for key in optionnal_output_dict:
                        sic4dvar_dict[key] = optionnal_output_dict[key]
            else:
                node_z, node_w, optionnal_output_dict = Extrapolation(node_z=sic4dvar_dict['input_data']['node_z'], node_w=sic4dvar_dict['input_data']['node_w'], node_x=sic4dvar_dict['input_data']['node_x'], reach_t=sic4dvar_dict['input_data']['reach_t'], corx=corx, corx_array=corx_array, cort_wse=cort_wse, cort_width=cort_width, gnuplot_saving=sic4dvar_dict['param_dict']['gnuplot_saving'], reach_id=sic4dvar_dict['input_data']['reach_id'], output_dir=sic4dvar_dict['param_dict']['output_dir'], run_type=sic4dvar_dict['param_dict']['run_type'], use_large_deviations=params.large_deviations, start_from_downstream=params.start_from_downstream, run_preprocessing=params.run_preprocessing, run_extrapolation=params.run_extrapolation, pooling=params.pooling, regularization_profile=params.regularization_profile, use_quantiles_for_densification=params.use_quantiles_for_densification)
                if False:
                    sic4dvar_dict['input_data']['node_z'] = node_z[:, 0:node_z.shape[1] - 1]
                    sic4dvar_dict['input_data']['node_w'] = node_w[:, 0:node_w.shape[1] - 1]
                else:
                    sic4dvar_dict['input_data']['node_z'] = node_z
                    sic4dvar_dict['input_data']['node_w'] = node_w
                if params.save_quantiles_densification:
                    quantile_matrix = np.zeros((node_z.shape[0], 3))
                    for n in range(0, node_z.shape[0]):
                        quantile_matrix[n, :] = np.nanquantile(node_z[n, :], [0.33, 0.66, 0.99])
                    sic4dvar_dict['output']['quantile_matrix'] = quantile_matrix
                for key in optionnal_output_dict:
                    sic4dvar_dict[key] = optionnal_output_dict[key]
        valid_z = np.sum(~np.isnan(sic4dvar_dict['input_data']['node_z']))
        valid_w = np.sum(~np.isnan(sic4dvar_dict['input_data']['node_w']))
        logging.info(f'After extrapolation: {valid_z} valid wse and {valid_w} valid width')
    logging.info('Finished extrapolating the data.')
    if sic4dvar_dict['data_is_useable']:
        if params.fill_nan_with_neighbours:
            import time
            debut = time.time()
            logging.info('Filling NaN values with neighbours.')
            sic4dvar_dict['input_data']['node_z'] = deepcopy(fill_nan_with_neighbours(sic4dvar_dict['input_data']['node_z']))
            sic4dvar_dict['input_data']['node_w'] = deepcopy(fill_nan_with_neighbours(sic4dvar_dict['input_data']['node_w']))
            end = time.time()
            logging.info(f'Filled NaN values with neighbours in {end - debut} seconds.')
        if params.compute_mean_elevation:
            mean_elevation_profile_array = compute_mean_elevation_profile(sic4dvar_dict['input_data']['node_z'], sic4dvar_dict['input_data']['reach_t'])
            sic4dvar_dict['output']['mean_elevation_profile'] = deepcopy(mean_elevation_profile_array)
        sic4dvar_dict['output']['node_z'] = deepcopy(sic4dvar_dict['input_data']['node_z'])
        sic4dvar_dict['output']['node_w'] = deepcopy(sic4dvar_dict['input_data']['node_w'])
        sic4dvar_dict['output']['node_t'] = deepcopy(sic4dvar_dict['input_data']['node_t'])
        if not params.exclude_invalid_overpasses:
            sic4dvar_dict['data_is_useable'], sic4dvar_dict['observed_nodes'], sic4dvar_dict['list_to_keep'], sic4dvar_dict['removed_indices'] = remove_unuseable_nodes(sic4dvar_dict['input_data']['node_z'], sic4dvar_dict['input_data']['node_w'])
        sic4dvar_dict['output']['observed_nodes'] = sic4dvar_dict['observed_nodes']
    if sic4dvar_dict['data_is_useable'] and np.array(sic4dvar_dict['list_to_keep']).size > 0:
        sic4dvar_dict = create_filtered_arrays(sic4dvar_dict)
    else:
        logging.debug('Not enough usable nodes or times to process reach.')
        sic4dvar_dict['valid'] = 0
        sic4dvar_dict['output']['stopped_stage'] = 'densification'
        sic4dvar_dict['output']['node_z'] = deepcopy(sic4dvar_dict['input_data']['node_z'])
        sic4dvar_dict['output']['node_w'] = deepcopy(sic4dvar_dict['input_data']['node_w'])
        sic4dvar_dict['output']['node_t'] = deepcopy(sic4dvar_dict['input_data']['node_t'])
    if params.exclude_invalid_overpasses and sic4dvar_dict['data_is_useable']:
        sic4dvar_dict['filtered_data']['node_z'] = deepcopy(sic4dvar_dict['input_data']['tmp_filtered_node_z'])
        sic4dvar_dict['filtered_data']['node_w'] = deepcopy(sic4dvar_dict['input_data']['tmp_filtered_node_w'])
    logging.info('Finished creating filtered data with useable nodes.')
    return sic4dvar_dict

def sic4dvar_set_prior(sic4dvar_dict):
    logging.info(f'Qwbm value: {sic4dvar_dict['input_data']['reach_qwbm']}')
    sic4dvar_dict['output']['yearly_mean_q'] = sic4dvar_dict['input_data']['reach_qwbm']
    if sic4dvar_dict['data_is_useable']:
        sic4dvar_dict, flag_qwbm = replace_prior(sic4dvar_dict)
        logging.info(f'Qwbm value after modification: {sic4dvar_dict['input_data']['reach_qwbm']}')
    else:
        flag_qwbm = False
        logging.info('Qwbm value not modified because data is not usable.')
    sic4dvar_dict['output']['reach_qwbm'] = sic4dvar_dict['input_data']['reach_qwbm']
    return (sic4dvar_dict, flag_qwbm)

def synchronize_records(sic4dvar_dict, params):
    if (sic4dvar_dict['param_dict']['q_prior_from_stations'] or params.q_prior_from_ML) and params.station_period_only:
        if sic4dvar_dict['param_dict']['q_prior_from_stations']:
            string = 'station_qt_2000'
        elif params.q_prior_from_ML:
            string = 'ML_qt_2000'
        reach_times = []
        for t in range(0, len(sic4dvar_dict['filtered_data']['reach_t'])):
            reach_times.append(sic4dvar_dict['filtered_data']['reach_t'][t] / 86400)
        reach_times_to_keep = []
        for t in range(0, len(reach_times)):
            if reach_times[t] >= sic4dvar_dict['input_data'][string].min() and reach_times[t] <= sic4dvar_dict['input_data'][string].max():
                reach_times_to_keep.append(reach_times[t])
        last_time_instant = np.max(reach_times_to_keep) * 86400
        time_indexes_to_keep = []
        seen = set()
        for val in reach_times_to_keep:
            close_matches = np.where(np.abs(np.array(reach_times) - val) <= 0.01)[0]
            for idx in close_matches:
                if idx not in seen:
                    seen.add(idx)
                    time_indexes_to_keep.append(idx)
    else:
        last_time_instant = np.max(sic4dvar_dict['filtered_data']['reach_t'])
        time_indexes_to_keep = np.arange(0, len(sic4dvar_dict['filtered_data']['reach_t']))
    return (last_time_instant, time_indexes_to_keep)

def sic4dvar_compute_discharge(sic4dvar_dict, params, flag_qwbm):
    if sic4dvar_dict['data_is_useable'] and flag_qwbm:
        logging.info('INFO: preparing to estimate discharge.')
        if params.densification:
            pass
        if not params.densification:
            last_time_instant, time_indexes_to_keep = synchronize_records(sic4dvar_dict, params)
            sic4dvar_dict['list_to_keep'] = sic4dvar_dict['list_to_keep'][time_indexes_to_keep]
            sic4dvar_dict['filtered_data']['node_z'] = sic4dvar_dict['filtered_data']['node_z'][:, time_indexes_to_keep]
            sic4dvar_dict['filtered_data']['node_w'] = sic4dvar_dict['filtered_data']['node_w'][:, time_indexes_to_keep]
            sic4dvar_dict['filtered_data']['reach_t'] = sic4dvar_dict['filtered_data']['reach_t'][time_indexes_to_keep]
            sic4dvar_dict['filtered_data']['q_monthly_series'] = None
            if sic4dvar_dict['input_data']['create_series_from_monthly_q']:
                if 'q_monthly_series' in sic4dvar_dict['input_data']:
                    sic4dvar_dict['filtered_data']['q_monthly_series'] = np.array(sic4dvar_dict['input_data']['q_monthly_series'])[sic4dvar_dict['list_to_keep']]
                    weights = []
                    for i in range(1, 13):
                        weights.append(make_mask_period(sic4dvar_dict['filtered_data']['reach_t'], [i], plot=False, window_scale=params.monthly_mask))
                    sic4dvar_dict['filtered_data']['monthly_weights'] = weights
            if sic4dvar_dict['param_dict']['run_type'] == 'seq':
                sic4dvar_dict['filtered_data']['reach_s'] = sic4dvar_dict['filtered_data']['reach_s'][time_indexes_to_keep]
            else:
                logging.error("Reach slope from SWOT is not available in set mode, can't keep it for algo3. Check your config file.")
            cort_slope = 0.0
            cort_tmp = []
            for t in range(0, len(sic4dvar_dict['filtered_data']['reach_t'])):
                cort_tmp.append(sic4dvar_dict['filtered_data']['reach_t'][t])
            sic4dvar_dict['cort_slope'] = np.mean(np.diff(cort_tmp))
            SLOPEM1, sic4dvar_dict = compute_slope(sic4dvar_dict, params)
            if sic4dvar_dict['reverse_order']:
                SLOPEM1 = -SLOPEM1
                if 'SLOPEM1_smoothed' in sic4dvar_dict['output']:
                    sic4dvar_dict['output']['SLOPEM1_smoothed'] = -sic4dvar_dict['output']['SLOPEM1_smoothed']
            sic4dvar_dict['output']['SLOPEM1'] = deepcopy(SLOPEM1)
            sic4dvar_dict, bathymetry_array, apr_array = compute_bathymetry(sic4dvar_dict, params, SLOPEM1)
            sic4dvar_dict['output']['width'] = sic4dvar_dict['input_data']['node_xr']
            sic4dvar_dict['output']['elevation'] = sic4dvar_dict['input_data']['node_yr']
            if 'reach_xr' in bathymetry_array:
                sic4dvar_dict['output']['reach_xr'] = bathymetry_array['reach_xr']
            if 'reach_yr' in bathymetry_array:
                sic4dvar_dict['output']['reach_yr'] = bathymetry_array['reach_yr']
            sic4dvar_dict['output']['stopped_stage'] = 'bathymetry'
            if sic4dvar_dict['param_dict']['gnuplot_saving']:
                nodes2 = (sic4dvar_dict['input_data']['node_x'] - sic4dvar_dict['input_data']['node_x'][0]) / 1000
                times2 = sic4dvar_dict['input_data']['reach_t'] / 3600 / 24
                reach_id = str(sic4dvar_dict['input_data']['reach_id'])
                reach_id = verify_name_length(reach_id)
                output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', str(reach_id))
                if not Path(output_path).is_dir():
                    Path(output_path).mkdir(parents=True, exist_ok=True)
            if sic4dvar_dict['output']['valid']:
                logging.info(f'Computed bathymetry.')
            else:
                return sic4dvar_dict
            logging.info(f'Lauching discharge estimation.')
            if params.use_friction_external_file:
                friction_file = getattr(params, 'friction_external_file', '')
                if friction_file == '':
                    logging.warning('Friction external file option is True but no file provided.')
                    friction_values = None
                    friction_dates = None
                    friction_t = None
                else:
                    friction_values, friction_dates, friction_t = get_external_friction_data(friction_file, sic4dvar_dict['input_data']['reach_id'])
                friction_mean = 0
                time_scaling = 0
                for t in range(1, len(friction_t)):
                    friction_mean += (friction_values[t] + friction_values[t - 1]) / 2 * (friction_t[t] - friction_t[t - 1])
                    time_scaling += np.array(friction_t[t] - friction_t[t - 1])
                if time_scaling > 0:
                    friction_mean = friction_mean / time_scaling
                else:
                    logging.warning("Time scaling for friction external data is 0, can't compute mean friction.")
                    friction_mean = None
                if friction_mean is not None:
                    friction_mean = 1 / friction_mean
                sic4dvar_dict['input_data']['friction_value'] = friction_mean
            else:
                sic4dvar_dict['input_data']['friction_value'] = None
            series_mean = None
            monthly_time_series = None
            if sic4dvar_dict['input_data']['create_series_from_monthly_q']:
                series_mean = sic4dvar_dict['input_data']['series_mean']
                monthly_time_series = sic4dvar_dict['filtered_data']['q_monthly_series']
            cv_to_use = None
            if params.use_series_cv:
                cv_to_use = sic4dvar_dict['input_data']['series_cv']
            monthly_weights = None
            if params.use_flow_duration_q:
                if sic4dvar_dict['input_data']['create_series_from_monthly_q']:
                    monthly_weights = sic4dvar_dict['filtered_data']['monthly_weights']
            if sic4dvar_dict['param_dict']['gnuplot_saving']:
                reach_id = str(reach_id)
                reach_id = verify_name_length(reach_id)
                node_x = sic4dvar_dict['filtered_data']['node_x']
                test_t_array = sic4dvar_dict['filtered_data']['reach_t']
                nodes2 = (node_x - node_x[0]) / 1000
                times2 = test_t_array / 3600 / 24
                output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', str(reach_id))
                if not Path(output_path).is_dir():
                    Path(output_path).mkdir(parents=True, exist_ok=True)
                if params.create_series_from_monthly_q:
                    gnuplot_save_q(sic4dvar_dict['filtered_data']['q_monthly_series'], times2, output_path.joinpath('monthly_series'))
            sic4dvar_dict['output']['correlation_monthly'] = None
            sic4dvar_dict['output']['correlation_dynamic_slope'] = None
            if params.create_series_from_monthly_q:
                if 'q_monthly_series' in sic4dvar_dict['filtered_data']:
                    from lib.lib_indicators import pearson_correlation, integrated_mean
                    elevation_integral_space = np.zeros(len(sic4dvar_dict['filtered_data']['reach_t']))
                    for t in range(0, len(sic4dvar_dict['filtered_data']['reach_t'])):
                        elevation_integral_space[t] = integrated_mean(sic4dvar_dict['filtered_data']['node_z'][:, t], sic4dvar_dict['filtered_data']['node_x'])
                    if sic4dvar_dict['filtered_data']['q_monthly_series'] is None:
                        logging.warning("Monthly series is None, can't compute correlation with elevation integral space.")
                        sic4dvar_dict['output']['correlation_monthly'] = None
                        sic4dvar_dict['output']['correlation_dynamic_slope'] = None
                    else:
                        sic4dvar_dict['output']['correlation_monthly'], _ = pearson_correlation(sic4dvar_dict['filtered_data']['q_monthly_series'], elevation_integral_space)
                        dynamic_slope_criteria = sic4dvar_dict['output']['SLOPEM1_smoothed'] / np.nanmean(sic4dvar_dict['output']['SLOPEM1_smoothed']) - 1.0
                        elevation_criteria = elevation_integral_space / np.nanmean(elevation_integral_space) - 1.0
                        sic4dvar_dict['output']['correlation_dynamic_slope'], _ = pearson_correlation(dynamic_slope_criteria, elevation_criteria)
            if False:
                from matplotlib import pyplot as plt
                matrix_L4 = sic4dvar_dict['filtered_data']['node_z']
                x = np.arange(matrix_L4.shape[1])
                y = np.arange(matrix_L4.shape[0])
                X, Y = np.meshgrid(x, y)
                fig = plt.figure(figsize=(14, 6))
                ax1 = fig.add_subplot(121, projection='3d')
                ax1.plot_surface(X, Y, matrix_L4, cmap='viridis')
                ax1.set_title('Matrice L4')
                ax1.set_xlabel('Temps')
                ax1.set_ylabel('Section')
                ax1.set_zlabel('Valeur')
                ax1.view_init(elev=66, azim=-120)
                plt.tight_layout()
                plt.savefig('matrix_L4.png')
                plt.clf()
                print(bug)
            sic4dvar_dict['output']['q_algo31'], sic4dvar_dict['output']['valid'], sic4dvar_dict['reliability'], sic4dvar_dict['output']['Kmi_acc'], sic4dvar_dict['output']['Zb_acc'] = Z3(apr_array, sic4dvar_dict['input_data']['reach_qwbm'], params, SLOPEM1, sic4dvar_dict['output']['valid'], sic4dvar_dict['filtered_data']['node_z'], sic4dvar_dict['input_data']['node_z'], sic4dvar_dict['input_data']['node_z_ini'], sic4dvar_dict['filtered_data']['node_x'], sic4dvar_dict['last_node_for_integral'], bathymetry_array, sic4dvar_dict['param_dict'], sic4dvar_dict['input_data']['reach_id'], sic4dvar_dict['filtered_data']['reach_t'], bb=sic4dvar_dict['bb'], reliability=sic4dvar_dict['reliability'], Qsdev=sic4dvar_dict['input_data']['reach_qsdev'], last_time_instant=last_time_instant, input_data=sic4dvar_dict['input_data'], time_indexes_to_keep=time_indexes_to_keep, friction_value=sic4dvar_dict['input_data']['friction_value'], series_cv=cv_to_use, dynamic_slope=sic4dvar_dict['output']['SLOPEM1_smoothed'], monthly_time_series=monthly_time_series, monthly_weights=monthly_weights, series_mean=series_mean, correlation_criteria=sic4dvar_dict['output']['correlation_dynamic_slope'], sic4dvar_dict=sic4dvar_dict)
            print('orig Q31:', sic4dvar_dict['output']['q_algo31'], len(sic4dvar_dict['output']['q_algo31']))
            Q_EST_ARRAY = []
            Wmin = 10000
            Wmean = []
            for i in range(len(apr_array['node_w_simp'])):
                Wmean.append(min(apr_array['node_w_simp'][i]))
                if np.min(apr_array['node_w_simp'][i]) < Wmin:
                    Wmin = np.min(apr_array['node_w_simp'][i])
            Wmean = np.average(Wmean)
            Zb = np.zeros(len(sic4dvar_dict['filtered_data']['node_z']))
            for t in range(0, sic4dvar_dict['filtered_data']['node_z'].shape[1]):
                ZM = 1.0
                ZB_update = sic4dvar_dict['output']['Zb_acc']
                Zb2 = params.algo_bounds[1][1]
                Q_EST, _, Zb, _ = Z1(bathymetry_array['node_xr'], bathymetry_array['node_yr'], apr_array['node_a'], apr_array['node_p'], Wmean, sic4dvar_dict['last_node_for_integral'], Zb2, ZB_update, Zb, t, sic4dvar_dict['filtered_data']['node_x'], SLOPEM1, ZM=ZM, KMI=sic4dvar_dict['output']['Kmi_acc'], option_recompute_area=False, sic4dvar_dict=sic4dvar_dict)
                Q_EST_ARRAY.append(Q_EST[0])
            sic4dvar_dict['output']['Kmi_acc'] = sic4dvar_dict['output']['Kmi_acc'] / np.mean(Q_EST_ARRAY) * sic4dvar_dict['input_data']['reach_qwbm']
            if params.discharge_correction:
                sic4dvar_dict['output']['q_algo31'] = deepcopy(sic4dvar_dict['output']['q_algo31'] / np.nanmean(sic4dvar_dict['output']['q_algo31']) * sic4dvar_dict['input_data']['reach_qwbm'])
            if params.final_smoothing_discharge:
                Q31_2D = []
                for n in range(0, len(sic4dvar_dict['filtered_data']['node_z'])):
                    Q31_2D.append(sic4dvar_dict['output']['q_algo31'])
                Q31_2D = np.array(Q31_2D)
                from sic4dvar_functions.F446 import v
                Q31_2D = v(dim=0, value0_array=Q31_2D, base0_array=sic4dvar_dict['filtered_data']['reach_t'], max_iter=1, cor=3 * 24 * 3600, behavior='', inter_behavior=False, inter_behavior_min_thr=params.def_float_atol, inter_behavior_max_thr=params.DX_max_in, check_behavior='', min_change_v_thr=0.0001, plot=False, plot_title='Relaxation sweep in time Q31 without Interchange', clean_run=True, debug_mode=False, bias_correction_type=True, minimum_value=1.0)
                Q31_2D = np.array(Q31_2D[0])
                sic4dvar_dict['output']['q_algo31'] = deepcopy(Q31_2D[0])
            print('smoothed Q31:', sic4dvar_dict['output']['q_algo31'], len(sic4dvar_dict['output']['q_algo31']))
            if sic4dvar_dict['output']['valid']:
                logging.info(f'Estimated discharge with algo3.')
                sic4dvar_dict['output']['stopped_stage'] = 'discharge'
            if sic4dvar_dict['output']['Zb_acc'] is not np.nan:
                sic4dvar_dict['output']['z_bed'] = compute_z_bed(apr_array['node_w_simp'], sic4dvar_dict['filtered_data']['node_z'], bathymetry_array['node_xr'], bathymetry_array['node_yr'], sic4dvar_dict['output']['Zb_acc'])
                logging.debug('Z_bed: %s' % sic4dvar_dict['output']['z_bed'])
            sic4dvar_dict['output']['apr_array'] = apr_array
            if sic4dvar_dict['param_dict']['gnuplot_saving']:
                reach_id = str(reach_id)
                reach_id = verify_name_length(reach_id)
                node_x = sic4dvar_dict['filtered_data']['node_x']
                test_t_array = sic4dvar_dict['filtered_data']['reach_t']
                nodes2 = (node_x - node_x[0]) / 1000
                times2 = test_t_array / 3600 / 24
                output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', str(reach_id))
                if not Path(output_path).is_dir():
                    Path(output_path).mkdir(parents=True, exist_ok=True)
                gnuplot_save_q(sic4dvar_dict['output']['q_algo31'], times2, output_path.joinpath('q31_out'))
            if params.experimental_discharge_ML:
                ZB_update = sic4dvar_dict['output']['Zb_acc']
                Zb = np.zeros(len(sic4dvar_dict['filtered_data']['node_z']))
                Zb2 = params.algo_bounds[1][1]
                Wmin = 10000
                Wmean = []
                for i in range(len(apr_array['node_w_simp'])):
                    Wmean.append(min(apr_array['node_w_simp'][i]))
                    if np.min(apr_array['node_w_simp'][i]) < Wmin:
                        Wmin = np.min(apr_array['node_w_simp'][i])
                Wmean = np.average(Wmean)
                discharge_estimate = sic4dvar_dict['output']['q_algo31']
                time_instants = np.ma.masked_values(np.array(sic4dvar_dict['filtered_data']['reach_t']), value=-9999.0)
                if discharge_estimate.mask.size == 1:
                    valid_idx1 = np.where(discharge_estimate)[0]
                else:
                    valid_idx1 = np.where(np.isfinite(discharge_estimate))
                if time_instants.mask.size == 1:
                    valid_idx2 = np.where(time_instants)[0]
                else:
                    valid_idx2 = np.where(np.isfinite(time_instants))
                valid_idx3 = np.intersect1d(valid_idx1, valid_idx2)
                if discharge_estimate.mask.size == 1:
                    valid_idx1 = np.where(discharge_estimate)[0]
                else:
                    valid_idx1 = np.where(discharge_estimate.mask == False)
                if time_instants.mask.size == 1:
                    valid_idx1 = np.where(time_instants)[0]
                else:
                    valid_idx2 = np.where(time_instants.mask == False)
                valid_idx4 = np.intersect1d(valid_idx1, valid_idx2)
                valid_idx = np.intersect1d(valid_idx3, valid_idx4)
                time_instants = time_instants / 86400
                sic4dvar_date = daynum_to_date(time_instants[valid_idx], '2000-01-01')
                sic4dvar_df = pd.DataFrame({'sic4dvar_q': discharge_estimate[valid_idx], 'sic4dvar_date': sic4dvar_date})
                sic4dvar_df['date_only'] = pd.to_datetime(sic4dvar_df['sic4dvar_date'], format='%Y-%m-%d').dt.date
                sic4dvar_df['sic4dvar_qt'] = time_instants[valid_idx]
                qa31_t = deepcopy(np.array(sic4dvar_df['sic4dvar_qt']))
                if not params.force_specific_dates:
                    start_date = sic4dvar_df['date_only'].min()
                    end_date = sic4dvar_df['date_only'].max()
                else:
                    start_date = sic4dvar_df['date_only'].min()
                    end_date = sic4dvar_df['date_only'].max()
                ML_df = sic4dvar_dict['input_data']['ML_df']
                filtered_ML_df = ML_df[(ML_df['date_only'] >= start_date) & (ML_df['date_only'] <= end_date)]
                start_date_ML = filtered_ML_df['date_only'].min()
                end_date_ML = filtered_ML_df['date_only'].max()
                times_ML = filtered_ML_df[(filtered_ML_df['date_only'] >= start_date) & (filtered_ML_df['date_only'] <= end_date)]
                times_sic = sic4dvar_df[(sic4dvar_df['date_only'] >= start_date) & (sic4dvar_df['date_only'] <= end_date)]
                time_system = times_sic['sic4dvar_qt']
                if len(time_system) < 2:
                    logging.info("Time system size < 2, can't compute integrated estimated mean.")
                    return {}
                q_ref = []
                q_est = []
                t_ref = []
                q_ref_ML = []
                for t in range(0, len(time_system)):
                    q_ref_ML.append(interp_pdf_tables(len(filtered_ML_df['station_q']) - 1, time_system.iloc[t], np.array(filtered_ML_df['station_qt']), np.array(filtered_ML_df['station_q'])))
                    q_est.append(sic4dvar_df['sic4dvar_q'][t])
                    t_ref.append(time_system.iloc[t])
                ZM = 1.0
                ZM_opt_array = []
                for t in range(0, len(t_ref)):
                    Q_REF = q_ref_ML[t]
                    from sic4dvar_algos.X653 import Z2
                    ZM_opt = minimize_scalar(Z2, bounds=(0.33, 3.0), method='bounded', options={'maxiter': 10}).x
                    ZM_opt_array.append(ZM_opt)
                ZM_opt_array_new = np.array([float(x) for x in ZM_opt_array])
                if sic4dvar_dict['param_dict']['gnuplot_saving']:
                    reach_id = str(reach_id)
                    reach_id = verify_name_length(reach_id)
                    node_x = sic4dvar_dict['filtered_data']['node_x']
                    test_t_array = sic4dvar_dict['filtered_data']['reach_t']
                    nodes2 = (node_x - node_x[0]) / 1000
                    times2 = test_t_array / 3600 / 24
                    output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', str(reach_id))
                    if not Path(output_path).is_dir():
                        Path(output_path).mkdir(parents=True, exist_ok=True)
                    gnuplot_save_q(ZM_opt_array_new, times2, output_path.joinpath('ZM'))
                sigm0 = 0.2 * np.sqrt(100 / Wmean)
                ss = 0.0
                ss1 = 0.0
                for t in range(0, len(t_ref) - 1):
                    for n in range(0, len(sic4dvar_dict['filtered_data']['node_z'])):
                        ss += (((ZM_opt_array_new[t] - 1.0) * sic4dvar_dict['filtered_data']['node_z'][n, t]) ** 2 + ((ZM_opt_array_new[t + 1] - 1.0) * sic4dvar_dict['filtered_data']['node_z'][n, t + 1]) ** 2 / 2.0) * (t_ref[t + 1] - t_ref[t])
                        ss1 += t_ref[t + 1] - t_ref[t]
                alph0 = np.sqrt(ss1 * sigm0 ** 2 / ss)
                ZMS1 = np.zeros(len(t_ref))
                for t in range(0, len(t_ref)):
                    ss = 0.0
                    ss1 = 0.0
                    for n in range(0, len(sic4dvar_dict['filtered_data']['node_z'])):
                        ss += sic4dvar_dict['filtered_data']['node_z'][n, t]
                        ss1 += 1.0
                    ZMS1[t] = ss / ss1
                ss = 0.0
                ss1 = 0.0
                for t in range(0, len(t_ref) - 1):
                    ss += (((ZM_opt_array_new[t] - 1.0) * ZMS1[t]) ** 2 + ((ZM_opt_array_new[t + 1] - 1.0) * ZMS1[t + 1]) ** 2) / 2.0 * (t_ref[t + 1] - t_ref[t])
                    ss1 += t_ref[t + 1] - t_ref[t]
                alph1 = np.sqrt(ss1 * sigm0 ** 2 / ss)
                if alph1 > 1.0:
                    alph1 = 1.0
                sic4dvar_dict['output']['alph1'] = alph1
                for t in range(0, len(t_ref)):
                    ss = ZM_opt_array_new[t]
                    ZMS1[t] = 1.0 + alph1 * (ss - 1.0)
                ZM_opt_array_new = deepcopy(ZMS1)
                if sic4dvar_dict['param_dict']['gnuplot_saving']:
                    reach_id = str(reach_id)
                    reach_id = verify_name_length(reach_id)
                    node_x = sic4dvar_dict['filtered_data']['node_x']
                    test_t_array = sic4dvar_dict['filtered_data']['reach_t']
                    nodes2 = (node_x - node_x[0]) / 1000
                    times2 = test_t_array / 3600 / 24
                    output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', str(reach_id))
                    if not Path(output_path).is_dir():
                        Path(output_path).mkdir(parents=True, exist_ok=True)
                    gnuplot_save_q(ZM_opt_array_new, times2, output_path.joinpath('ZM2'))
                node_a2, node_p2, node_r2, node_w_simp2, _ = call_func_APR(sic4dvar_dict['filtered_data']['node_w'], sic4dvar_dict['filtered_data']['node_z'], sic4dvar_dict['input_data']['node_xr'], sic4dvar_dict['input_data']['node_yr'], params, sic4dvar_dict['param_dict'], coeff_array=ZM_opt_array_new)
                apr_array['node_a'] = deepcopy(node_a2)
                apr_array['node_p'] = deepcopy(node_p2)
                apr_array['node_r'] = deepcopy(node_r2)
                apr_array['node_w_simp'] = deepcopy(node_w_simp2)
                Q_EST_ARRAY = []
                if params.experimental_run_MMI:
                    for t in range(0, len(t_ref)):
                        ZM = ZM_opt_array_new[t]
                        Q_EST, _, _ = Z1(bathymetry_array['node_xr'], bathymetry_array['node_yr'], apr_array['node_a'], apr_array['node_p'], Wmean, sic4dvar_dict['last_node_for_integral'], Zb2, ZB_update, Zb, t, sic4dvar_dict['filtered_data']['node_x'], SLOPEM1, ZM=ZM, KMI=sic4dvar_dict['output']['Kmi_acc'], option_recompute_area=False, sic4dvar_dict=sic4dvar_dict)
                        Q_EST_ARRAY.append(Q_EST[0])
                else:
                    Q_EST_ARRAY, sic4dvar_dict['output']['valid'], sic4dvar_dict['reliability'], sic4dvar_dict['output']['Kmi_acc'], sic4dvar_dict['output']['Zb_acc'] = Z3(apr_array, sic4dvar_dict['input_data']['reach_qwbm'], params, SLOPEM1, sic4dvar_dict['output']['valid'], sic4dvar_dict['filtered_data']['node_z'], sic4dvar_dict['input_data']['node_z'], sic4dvar_dict['input_data']['node_z_ini'], sic4dvar_dict['filtered_data']['node_x'], sic4dvar_dict['last_node_for_integral'], bathymetry_array, sic4dvar_dict['param_dict'], sic4dvar_dict['input_data']['reach_id'], sic4dvar_dict['filtered_data']['reach_t'], bb=sic4dvar_dict['bb'], reliability=sic4dvar_dict['reliability'], Qsdev=sic4dvar_dict['input_data']['reach_qsdev'], last_time_instant=last_time_instant, input_data=sic4dvar_dict['input_data'], time_indexes_to_keep=time_indexes_to_keep, ZM_array=ZM_opt_array_new)
                Q_EST_ARRAY = np.array(Q_EST_ARRAY)
                sic4dvar_dict['output']['q_algo31'] = deepcopy(Q_EST_ARRAY)
                if True:
                    ss = 0.0
                    ss1 = 0.0
                    for t in range(0, len(Q_EST_ARRAY) - 1):
                        ss += (sic4dvar_dict['output']['q_algo31'][t + 1] + sic4dvar_dict['output']['q_algo31'][t]) / 2 * (t_ref[t + 1] - t_ref[t])
                        ss1 += t_ref[t + 1] - t_ref[t]
                    q_mean = ss / ss1
                    sic4dvar_dict['output']['q_algo31'] = deepcopy(sic4dvar_dict['output']['q_algo31'] * (sic4dvar_dict['input_data']['reach_qwbm'] / q_mean))
            if sic4dvar_dict['param_dict']['gnuplot_saving']:
                reach_id = str(reach_id)
                reach_id = verify_name_length(reach_id)
                node_x = sic4dvar_dict['filtered_data']['node_x']
                test_t_array = sic4dvar_dict['filtered_data']['reach_t']
                nodes2 = (node_x - node_x[0]) / 1000
                times2 = test_t_array / 3600 / 24
                output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', str(reach_id))
                if not Path(output_path).is_dir():
                    Path(output_path).mkdir(parents=True, exist_ok=True)
                gnuplot_save_q(sic4dvar_dict['output']['q_algo31'], times2, output_path.joinpath('q31_out_2'))
                output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', str(reach_id))
                if not Path(output_path).is_dir():
                    Path(output_path).mkdir(parents=True, exist_ok=True)
            logging.info(f'Finished discharge estimation.')
        Q31 = sic4dvar_dict['output']['q_algo31']
        if np.array(Q31).size == 0:
            sic4dvar_dict['output']['valid'] = 0
            return sic4dvar_dict
        logging.debug('Discharge estimation complete. Q31: %s' % Q31)
        if sic4dvar_dict['param_dict']['gnuplot_saving']:
            reach_id = str(reach_id)
            reach_id = verify_name_length(reach_id)
            reach_id = verify_name_length(reach_id)
            node_x = node_x
            test_t_array = sic4dvar_dict['filtered_data']['reach_t']
            nodes2 = (node_x - node_x[0]) / 1000
            times2 = test_t_array / 3600 / 24
            output_dir = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', reach_id)
            if not output_dir.is_dir():
                output_dir.mkdir(parents=True, exist_ok=True)
            if not output_dir.is_dir():
                output_dir.mkdir(parents=True, exist_ok=True)
            gnuplot_save_q(Q31, times2, output_dir.joinpath('qalgo31_final'))
        sic4dvar_dict['output']['reach_length'] = sic4dvar_dict['input_data']['node_x'].max() - sic4dvar_dict['input_data']['node_x'].min()
        reach_slope = sic4dvar_dict['output']['SLOPEM1'] / sic4dvar_dict['output']['reach_length']
        sic4dvar_dict['output']['reach_slope'] = np.full(sic4dvar_dict['input_data']['reach_t'].size, np.nan)
        sic4dvar_dict['output']['reach_slope'][sic4dvar_dict['list_to_keep']] = reach_slope
        sic4dvar_dict['output']['q_algo31'] = Q31
        sic4dvar_dict['output']['q_algo31'] = build_output_q_masked_array(sic4dvar_dict, 'q_algo31')
        sic4dvar_dict['output']['time'] = build_output_q_masked_array(sic4dvar_dict, 'time')
        if 'apr_array' in sic4dvar_dict['output']:
            sic4dvar_dict['output']['apr_array']['node_a'] = build_output_2d_masked_array(sic4dvar_dict, sic4dvar_dict['output']['apr_array']['node_a'])
            sic4dvar_dict['output']['apr_array']['node_r'] = build_output_2d_masked_array(sic4dvar_dict, sic4dvar_dict['output']['apr_array']['node_r'])
        sic4dvar_dict['output']['width'] = sic4dvar_dict['input_data']['node_xr']
        sic4dvar_dict['output']['elevation'] = sic4dvar_dict['input_data']['node_yr']
        if sic4dvar_dict['param_dict']['run_type'] == 'set':
            NBR_REACHES = len(sic4dvar_dict['filtered_data']['reach_w'])
        if sic4dvar_dict['param_dict']['run_type'] == 'seq':
            NBR_REACHES = 1
        if sic4dvar_dict['param_dict']['run_algo5']:
            sic4dvar_dict = launch_algo5(sic4dvar_dict, NBR_REACHES)
            logging.info('Finished running Algo5.')
    else:
        logging.warning('Not enough consecutive data available to process reach.')
        sic4dvar_dict['output']['valid'] = 0
    return sic4dvar_dict
