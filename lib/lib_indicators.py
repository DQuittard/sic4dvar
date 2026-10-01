"""
SIC4DVAR-LC
Copyright (C) 2025 INRAE

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
from scipy.stats import spearmanr, pearsonr, rankdata
from lib.lib_dates import daynum_to_date, check_date_format
import numpy as np
import datetime
import re
import logging
import pandas as pd

def pearson_correlation(y_true, y_pred, time_frequency=np.array([])):
    n = len(y_true)
    if time_frequency.any() == 0:
        stat, pvalue = pearsonr(y_true, y_pred)
    else:
        y_true_mean = np.average(y_true, weights=time_frequency)
        y_pred_mean = np.average(y_pred, weights=time_frequency)
        ssd_true = np.sqrt(1 / (n - 1) * np.nansum(time_frequency * (y_true - y_true_mean) ** 2))
        ssd_pred = np.sqrt(1 / (n - 1) * np.nansum(time_frequency * (y_pred - y_pred_mean) ** 2))
        pvalue = None
        stat = 1 / (n - 1) * np.nansum(time_frequency * ((y_true - y_true_mean) / ssd_true) * ((y_pred - y_pred_mean) / ssd_pred))
    return (stat, pvalue)

def spearman_correlation(y_true, y_pred):
    return spearmanr(y_true, y_pred)

def spearman_correlation_coded(y_true, y_pred):
    ranked_y_true = rankdata(y_true)
    ranked_y_pred = rankdata(y_pred)
    diff = rankdata(y_true) - rankdata(y_pred)
    n = len(y_true)
    rs = 1.0 - 6 * np.nansum(diff ** 2) / (n * (n ** 2 - 1))
    return rs

def spearman_correlation_coded2(y_true, y_pred):
    ranked_y_true = rankdata(y_true)
    ranked_y_pred = rankdata(y_pred)
    cov = np.cov(ranked_y_true, ranked_y_pred, bias=True)[0][1]
    rs = cov / (np.std(ranked_y_true) * np.std(ranked_y_pred))
    return rs

def nse(y_true, y_pred, time_frequency=np.array([])):
    if time_frequency.any():
        nse = 1 - np.nansum((y_true - y_pred) ** 2 * time_frequency) / np.nansum((y_true - np.average(y_true, weights=time_frequency)) ** 2 * time_frequency)
    else:
        nse = 1 - np.nansum((y_true - y_pred) ** 2) / np.nansum((y_true - np.nanmean(y_true)) ** 2)
    return nse

def nse1(y_true, y_pred, time_frequency=np.array([])):
    if time_frequency.any():
        nse1 = 1 - np.nansum(abs(y_true - y_pred) * time_frequency) / np.nansum(abs(y_true - np.average(y_true, weights=time_frequency)) * time_frequency)
    else:
        nse1 = 1 - np.nansum(abs(y_true - y_pred)) / np.nansum(abs(y_true - np.nanmean(y_true)))
    return nse1

def nrmse(y_true, y_pred, time_frequency=np.array([])):
    n = len(y_true)
    if time_frequency.any():
        RMSE = rmse(y_true, y_pred, time_frequency)
        y_true_mean = np.average(y_true, weights=time_frequency)
    else:
        RMSE = rmse(y_true, y_pred)
        y_true_mean = np.nanmean(y_true)
    return RMSE / y_true_mean

def rmse(y_true, y_pred, time_frequency=np.array([])):
    if (np.isfinite(y_true) != np.isfinite(y_pred)).any():
        valid_idx = np.logical_and(np.isfinite(y_true), np.isfinite(y_pred))
        y_true = y_true[valid_idx]
        y_pred = y_pred[valid_idx]
    n = len(y_true)
    if time_frequency.any():
        RMSE = np.sqrt(np.average((y_true - y_pred) ** 2, weights=time_frequency))
    else:
        RMSE = np.sqrt(np.nansum((y_true - y_pred) ** 2) / n)
    return RMSE

def extrema_high(y_true, y_pred, time_frequency):
    percen_list = [0.99, 0.98, 0.97, 0.96, 0.95]
    y_true_percent_values_list = [np.percentile(y_true, p) for p in percen_list]
    y_pred_percent_values_list = [np.percentile(y_pred, p) for p in percen_list]
    return (np.mean(y_true_percent_values_list) - np.mean(y_pred_percent_values_list)) / np.mean(y_true_percent_values_list)

def extrema_low(y_true, y_pred, time_frequency):
    percen_list = [0.01, 0.02, 0.03, 0.04, 0.05]
    y_true_percent_values_list = [np.percentile(y_true, p) for p in percen_list]
    y_pred_percent_values_list = [np.percentile(y_pred, p) for p in percen_list]
    return (np.mean(y_true_percent_values_list) - np.mean(y_pred_percent_values_list)) / np.mean(y_true_percent_values_list)

def log_cosh_loss(y_true, y_pred, bias=0.0):
    """Calculates Log-Cosh Loss."""
    error = y_pred - y_true - bias
    return np.mean(np.log(np.cosh(error)))

def tweedie_loss(y_true, y_pred, power):
    """Calculates Tweedie loss."""
    if power <= 0 or power >= 1:
        if power == 0:
            return (y_true - y_pred) ** 2
        elif power == 1:
            return 2 * (y_true * np.log(y_true / y_pred) - (y_true - y_pred))
        elif power == 2:
            return 2 * (np.log(y_pred / y_true) + y_true / y_pred - 1)
    print('Tweedie for 1 < p < 2 is complex; use a library.')
    return None

def bias(y_true, y_pred, time_frequency=np.array([])):
    if (np.isfinite(y_true) != np.isfinite(y_pred)).any():
        valid_idx = np.logical_and(np.isfinite(y_true), np.isfinite(y_pred))
        y_true = y_true[valid_idx]
        y_pred = y_pred[valid_idx]
    if time_frequency.any():
        y_true_mean = np.nanmean(y_true * time_frequency)
        y_pred_mean = np.nanmean(y_pred * time_frequency)
    else:
        y_true_mean = np.nanmean(y_true)
        y_pred_mean = np.nanmean(y_pred)
    bias = y_pred_mean - y_true_mean
    return bias

def nbias(y_true, y_pred, time_frequency=np.array([])):
    if (np.isfinite(y_true) != np.isfinite(y_pred)).any():
        valid_idx = np.logical_and(np.isfinite(y_true), np.isfinite(y_pred))
        y_true = y_true[valid_idx]
        y_pred = y_pred[valid_idx]
    if time_frequency.any():
        y_true_mean = np.nanmean(y_true * time_frequency)
    else:
        y_true_mean = np.nanmean(y_true)
    nbias = bias(y_true, y_pred, time_frequency) / y_true_mean
    return nbias

def absnbias(y_true, y_pred, time_frequency=np.array([])):
    abs_nbias = abs(nbias(y_true, y_pred, time_frequency))
    return abs_nbias

def compute_all_indicators_from_predict_true(y_true, y_pred, time_frequency=np.array([])):
    indicators = {}
    indicators['spearman'], *_ = spearman_correlation(y_true, y_pred)
    indicators['nrmse'] = nrmse(y_true, y_pred, time_frequency)
    indicators['nse'] = nse(y_true, y_pred, time_frequency)
    indicators['nnse'] = 1 / (2 - indicators['nse'])
    indicators['pearson'], *_ = pearson_correlation(y_true, y_pred, time_frequency)
    indicators['nbias'] = nbias(y_true, y_pred, time_frequency)
    indicators['absnbias'] = absnbias(y_true, y_pred, time_frequency)
    indicators['nse1'] = nse1(y_true, y_pred, time_frequency)
    indicators['extrema_high'] = extrema_high(y_true, y_pred, time_frequency)
    indicators['extrema_low'] = extrema_low(y_true, y_pred, time_frequency)
    return indicators

def parse_masked_array_text(text):
    cleaned = text.replace('...', ' ').replace('[', ' ').replace(']', ' ').replace(',', ' ')
    tokens = cleaned.split()
    values = []
    for token in tokens:
        if token == '--':
            values.append(np.nan)
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue
    return np.array(values, dtype=float)

def station_dates_from_qt_days(station_qt_days):
    epoch = datetime.datetime(1, 1, 1)
    return [epoch + datetime.timedelta(days=float(day)) for day in station_qt_days]

def parse_datetime_list_text(text):
    pattern = 'datetime\\.datetime\\((\\d+),\\s*(\\d+),\\s*(\\d+),\\s*(\\d+),\\s*(\\d+)\\)'
    matches = re.findall(pattern, text)
    return [datetime.datetime(int(year), int(month), int(day), int(hour), int(minute)) for year, month, day, hour, minute in matches]

def tmp_check_datetime_or_daynum(y_true_dim):
    y_true_dim_array = np.asarray(y_true_dim, dtype=object)
    if y_true_dim_array.size > 0 and isinstance(y_true_dim_array.flat[0], (datetime.datetime, datetime.date, np.datetime64)):
        epoch = datetime.datetime(1, 1, 1)
        converted_days = []
        for value in y_true_dim_array:
            if isinstance(value, np.datetime64):
                value = value.astype('datetime64[us]').astype(datetime.datetime)
            if isinstance(value, datetime.date) and (not isinstance(value, datetime.datetime)):
                value = datetime.datetime.combine(value, datetime.time.min)
            converted_days.append((value - epoch).total_seconds() / 86400.0)
        y_true_dim = np.array(converted_days, dtype=float)
    return y_true_dim

def run_indicators(reach_id='99999999999', y_true_val=None, y_true_dim=None, y_pred_val=None, y_pred_dim=None, min_dates=10, indicators_list=None, lite_run=True, other_sources={}, return_preprocessed_df=False):
    input_data = {}
    input_data['station_q'] = y_true_val
    y_true_dim = tmp_check_datetime_or_daynum(y_true_dim)
    input_data['station_qt'] = y_true_dim
    input_data['station_date'] = ''
    input_data['reach_t'] = [0.0, 0.0]
    input_data['reach_w_mean'] = [0.0, 0.0]
    input_data['reach_length'] = [0.0, 0.0]
    input_data['reach_slope'] = [0.0, 0.0]
    if other_sources != {}:
        input_data.update(other_sources)
    indicators = compute_indicators(input_data, y_pred_val, y_pred_dim, indicators_experimental_computation=True, force_specific_dates=False, reach_id=reach_id, ML_daily_comparison_experimental=False, min_dates=10, indicators_list=indicators_list, lite_run=lite_run, return_preprocessed_df=return_preprocessed_df)
    return indicators

def clean_invalid_values_for_input(discharge, time_instants):
    discharge = np.ma.asarray(discharge).ravel()
    time_instants = np.ma.asarray(time_instants).ravel()
    minimum_value = 0.0
    if discharge.size != time_instants.size:
        n_common = min(discharge.size, time_instants.size)
        logging.warning('Mismatched input sizes for cleaning (discharge=%s, time=%s). Truncating to %s.', discharge.size, time_instants.size, n_common)
        discharge = discharge[:n_common]
        time_instants = time_instants[:n_common]
    discharge_values = np.asarray(discharge.filled(np.nan), dtype=float)
    time_values = np.asarray(time_instants.filled(np.nan), dtype=float)
    discharge_mask = np.asarray(np.ma.getmaskarray(discharge), dtype=bool)
    time_mask = np.asarray(np.ma.getmaskarray(time_instants), dtype=bool)
    valid_mask = np.isfinite(discharge_values) & np.isfinite(time_values) & (discharge_values > minimum_value) & (time_values > minimum_value) & ~discharge_mask & ~time_mask
    return (discharge_values[valid_mask], time_values[valid_mask])

def clean_invalid_values_for_input_old(discharge, time_instants):
    if not np.ma.isMaskedArray(discharge):
        discharge = np.ma.masked_array(discharge, mask=[False] * len(discharge))
    if not np.ma.isMaskedArray(time_instants):
        time_instants = np.ma.masked_array(time_instants, mask=[False] * len(time_instants))
    if discharge.mask.size == 1:
        valid_idx1 = np.where(discharge)[0]
    else:
        valid_idx1 = np.where(np.isfinite(discharge))
    if time_instants.mask.size == 1:
        valid_idx2 = np.where(time_instants)[0]
    else:
        valid_idx2 = np.where(np.isfinite(time_instants))
    valid_idx3 = np.intersect1d(valid_idx1, valid_idx2)
    if discharge.mask.size == 1:
        valid_idx1 = np.where(discharge)[0]
    else:
        valid_idx1 = np.where(discharge.mask == False)
    if time_instants.mask.size == 1:
        valid_idx1 = np.where(time_instants)[0]
    else:
        valid_idx2 = np.where(time_instants.mask == False)
    valid_idx4 = np.intersect1d(valid_idx1, valid_idx2)
    valid_idx = np.intersect1d(valid_idx3, valid_idx4)
    return (discharge[valid_idx], time_instants[valid_idx])

def make_dataframe_from_input(discharge, time_instants, prefix_string, reference_date, data_date):
    discharge, time_instants = clean_invalid_values_for_input(discharge, time_instants)
    df = pd.DataFrame({f'{prefix_string}_q': discharge, f'{prefix_string}_times': time_instants})
    df[f'{prefix_string}_dates_string'] = daynum_to_date(df[f'{prefix_string}_times'], reference_date)
    df[f'{prefix_string}_datetimes'] = pd.to_datetime(df[f'{prefix_string}_dates_string'], format=check_date_format(reference_date))
    reference_datetime = datetime.datetime.strptime(reference_date, check_date_format(reference_date))
    data_datetime = datetime.datetime.strptime(data_date, check_date_format(data_date))
    delta_days = (data_datetime - reference_datetime).days
    df[f'{prefix_string}_times_in_days'] = [v - delta_days for v in df[f'{prefix_string}_times']]
    return df

def create_time_system(algo_df, station_df, ml_df, times_algo, input_data):
    from sic4dvar_functions.sic4dvar_helper_functions import interp_pdf_tables
    from copy import deepcopy
    q_ref = []
    q_est = []
    t_ref = []
    q_ref_ML = []
    time_system = deepcopy(times_algo)
    if len(time_system) < 2:
        logging.info("Time system size < 2, can't compute integrated estimated mean.")
        return ([], [], [], [])
    for t in range(0, len(time_system)):
        value_t = time_system.iloc[t]
        index_algo_to_use = np.where(algo_df['algo_times_in_days'] == value_t)[0]
        q_ref.append(interp_pdf_tables(len(station_df['station_q']) - 1, time_system.iloc[t], np.array(station_df['station_times_in_days']), np.array(station_df['station_q'])))
        if 'q_ML' in input_data.keys():
            q_ref_ML.append(interp_pdf_tables(len(ml_df['ML_q']) - 1, time_system.iloc[t], np.array(ml_df['ML_times_in_days']), np.array(ml_df['ML_q'])))
        q_est.append(algo_df['algo_q'].iloc[index_algo_to_use].values[0])
        t_ref.append(time_system.iloc[t])
    if False:
        q_est2 = []
        t_ref2 = []
        for t in range(0, len(time_system)):
            t_ref2.append(time_system.iloc[t])
        time_system = deepcopy(station_df['station_times_in_days'])
        t_ref3 = []
        q_ref2 = []
        for t in range(0, len(time_system)):
            q_ref2.append(station_df['station_q'].iloc[t])
            q_est2.append(interp_pdf_tables(len(algo_df['algo_q']) - 1, time_system.iloc[t], np.array(algo_df['algo_times_in_days']), np.array(algo_df['algo_q'])))
            t_ref3.append(time_system.iloc[t])
        from matplotlib import pyplot as plt
        plt.plot(t_ref3, q_est2, label='Predicted Q')
        plt.plot(t_ref3, q_ref2, label='Source Q')
        plt.legend()
        plt.savefig('/mnt/DATA/worksync/sic_tools/stage3.6_timesystem_for_each.png')
        plt.clf()
    if np.array(q_ref_ML).size == 0:
        q_ref_ML = [-9999.0] * len(q_ref)
    return (q_ref, q_est, t_ref, q_ref_ML)

def preprocess_input_data(input_data, algo_q, algo_t, force_specific_dates=False):
    station_df = make_dataframe_from_input(input_data['station_q'], input_data['station_qt'], 'station', '0001-01-01', '2000-01-01')
    algo_df = make_dataframe_from_input(algo_q, algo_t, 'algo', '2000-01-01', '2000-01-01')
    if 'q_ML' in input_data and 't_ML' in input_data:
        ml_df = make_dataframe_from_input(input_data['q_ML'], input_data['t_ML'], 'ML', '2000-01-01', '2000-01-01')
    else:
        ml_df = pd.DataFrame(columns=['ML_q', 'ML_times_in_days', 'ML_datetimes'])
    if not force_specific_dates:
        start_date_algo = algo_df['algo_datetimes'].min()
        end_date_algo = algo_df['algo_datetimes'].max()
    else:
        start_date_algo = algo_df['algo_datetimes'].min()
        end_date_algo = algo_df['algo_datetimes'].max()
    station_df = station_df[(station_df['station_datetimes'] >= start_date_algo) & (station_df['station_datetimes'] <= end_date_algo)]
    if station_df.empty:
        logging.warning("No data for station on SWOT observations period. Can't compute indicators.")
        data_df = pd.DataFrame()
    elif not station_df.empty:
        start_date_station = station_df['station_datetimes'].min()
        end_date_station = station_df['station_datetimes'].max()
        if start_date_station < start_date_algo:
            start_date = start_date_algo
        else:
            start_date = start_date_station
        if end_date_station > end_date_algo:
            end_date = end_date_algo
        else:
            end_date = end_date_station
        times_algo = algo_df[(algo_df['algo_datetimes'] >= start_date) & (algo_df['algo_datetimes'] <= end_date)]
        times_algo = times_algo['algo_times_in_days']
        q_ref, q_est, t_ref, q_ref_ML = create_time_system(algo_df, station_df, ml_df, times_algo, input_data)
        if len(q_ref) == 0 or len(q_est) == 0 or len(t_ref) == 0:
            logging.info('No valid time system after overlap filtering; returning empty dataframe.')
            data_df = pd.DataFrame()
        else:
            data_df = pd.DataFrame({'station_q': q_ref, 'time_in_days': t_ref, 'algo_q': q_est, 'ML_q': q_ref_ML})
            data_df = data_df[data_df['station_q'] > 0.0].reset_index(drop=True)
    return data_df

def integrated_mean(array, dimension):
    array_mean = 0.0
    time_scaling = 0.0
    for j in range(1, len(array)):
        array_mean += (array[j] + array[j - 1]) / 2 * (dimension[j] - dimension[j - 1])
        time_scaling += dimension[j] - dimension[j - 1]
    if time_scaling == 0:
        logging.error(f'time_scaling is zero in integrated mean !!')
    array_mean = array_mean / time_scaling
    return array_mean

def integrated_variance(array, dimension, array_mean):
    array_var = 0.0
    time_scaling = 0.0
    for j in range(1, len(array)):
        array_var += ((array[j] - array_mean) ** 2 + (array[j - 1] - array_mean) ** 2) / 2 * (dimension[j] - dimension[j - 1])
        time_scaling += dimension[j] - dimension[j - 1]
    array_var = np.sqrt(array_var / time_scaling)
    return array_var

def create_additional_dict(input_dict, prefix_string, data_df, input_value, input_value_array, Q_ref_mean, variance=0.0):
    input_dict[f'{prefix_string}'] = {}
    input_dict[f'{prefix_string}']['mean_value'] = input_value
    input_dict[f'{prefix_string}']['var_value'] = variance
    input_dict[f'{prefix_string}']['array_value'] = input_value_array
    input_dict[f'{prefix_string}']['cv'] = input_value / variance if variance != 0 else np.nan
    input_dict[f'{prefix_string}']['bias'] = input_value - Q_ref_mean
    input_dict[f'{prefix_string}']['nbias'] = (input_value - Q_ref_mean) / Q_ref_mean
    input_dict[f'{prefix_string}']['absnbias'] = np.abs(input_value - Q_ref_mean) / Q_ref_mean
    return input_dict

def interpolated_pearson(x, x_mean, y, y_mean, t, times):
    pearson_nominator = ((x[t] - x_mean) * (y[t] - y_mean) + (x[t - 1] - x_mean) * (y[t - 1] - y_mean)) / 2 * (times[t] - times[t - 1])
    pearson_d1 = ((x[t] - x_mean) ** 2 + (x[t - 1] - x_mean) ** 2) / 2 * (times[t] - times[t - 1])
    pearson_d2 = ((y[t] - y_mean) ** 2 + (y[t - 1] - y_mean) ** 2) / 2 * (times[t] - times[t - 1])
    return (pearson_nominator, pearson_d1, pearson_d2)

def compute_indicators(input_data, algo_q, algo_t, indicators_experimental_computation, force_specific_dates, reach_id, ML_daily_comparison_experimental=False, min_dates=10, indicators_list=None, lite_run=True, return_preprocessed_df=False):
    if indicators_list is None:
        indicators_list = ['mean_value', 'bias', 'nbias', 'absnbias', 'cosine_similarity', 'nrmse', 'nrmse2', 'log_cosh', 'tweedie_gamma', 'KGE_nrmse', 'KGE_cosh', 'KGE', 'pearson', 'spearman', 'cv', 'relative_error']
    if 'station_q' in input_data.keys() and 'station_date' in input_data.keys():
        logging.info('Indicators computation')
        if True:
            data_df = preprocess_input_data(input_data, algo_q, algo_t)
            if return_preprocessed_df:
                return data_df
            if not isinstance(data_df, pd.DataFrame) or data_df.empty:
                logging.info(f'No valid overlap after preprocessing for reach {reach_id}; skipping indicators.')
                return {}
            x = []
            option_linear_combination = False
            if option_linear_combination:
                if 'q_ML' in input_data.keys():
                    station_array = np.array(data_df['station_q'])
                    matrix_sic_ML = np.vstack([data_df['sic4dvar_q'], data_df['ML_q']]).T
                    x = np.linalg.lstsq(matrix_sic_ML, station_array)[0]
                    sum_x_coeff = x[0] + x[1]
                    x[0] = x[0] / sum_x_coeff
                    x[1] = x[1] / sum_x_coeff
                    approximated_discharge = x[0] * matrix_sic_ML[:, 0] + x[1] * matrix_sic_ML[:, 1]
                    data_df['sic4dvar_q_orig'] = data_df['sic4dvar_q']
                    data_df['sic4dvar_q'] = approximated_discharge
                else:
                    pass
            Q_est_mean = integrated_mean(np.array(data_df['algo_q']), np.array(data_df['time_in_days']))
            Q_est_var = integrated_variance(np.array(data_df['algo_q']), np.array(data_df['time_in_days']), Q_est_mean)
            Q_ref_mean = integrated_mean(np.array(data_df['station_q']), np.array(data_df['time_in_days']))
            Q_ref_var = integrated_variance(np.array(data_df['station_q']), np.array(data_df['time_in_days']), Q_ref_mean)
            if Q_ref_var < 1e-09:
                logging.warning('Variance is very small, removing reach from analysis for indicators computation. REACH_ID: ' + str(reach_id))
                return {}
            if 'q_ML' in input_data.keys():
                Q_ML_mean = integrated_mean(np.array(data_df['ML_q']), np.array(data_df['time_in_days']))
                Q_ML_var = integrated_variance(np.array(data_df['ML_q']), np.array(data_df['time_in_days']), Q_ML_mean)
                ML_coeff_variation = Q_ML_var / Q_ML_mean
            additional_indicators_dict = {}
            additional_indicators_dict = create_additional_dict(additional_indicators_dict, 'algo', data_df, Q_est_mean, data_df['algo_q'], Q_ref_mean, variance=Q_est_var)
            if not lite_run:
                additional_indicators_dict = create_additional_dict(additional_indicators_dict, 'truth', data_df, Q_ref_mean, data_df['station_q'], Q_ref_mean, variance=Q_ref_var)
            if 'qwbm_prior' in list(input_data.keys()):
                additional_indicators_dict = create_additional_dict(additional_indicators_dict, 'QWBM_prior', data_df, input_data['qwbm_prior'], np.full(len(data_df['station_q']), input_data['qwbm_prior']), Q_ref_mean)
            if 'grades_prior' in list(input_data.keys()):
                additional_indicators_dict = create_additional_dict(additional_indicators_dict, 'GRADES_prior', data_df, input_data['grades_prior'], np.full(len(data_df['station_q']), input_data['grades_prior']), Q_ref_mean)
            if 'ML_prior' in list(input_data.keys()):
                additional_indicators_dict = create_additional_dict(additional_indicators_dict, 'ML_prior', data_df, input_data['ML_prior'], np.full(len(data_df['station_q']), input_data['ML_prior']), Q_ref_mean)
            if 'q_ML' in input_data.keys():
                additional_indicators_dict = create_additional_dict(additional_indicators_dict, 'ML', data_df, Q_ML_mean, data_df['ML_q'], Q_ref_mean, variance=Q_ML_var)
            time_scaling = 0.0
            for source in additional_indicators_dict.keys():
                additional_indicators_dict[source]['nrmse_sum'] = 0.0
                additional_indicators_dict[source]['nrmse2_sum'] = 0.0
                additional_indicators_dict[source]['pearson_nom'] = 0.0
                additional_indicators_dict[source]['pearson_d1'] = 0.0
                additional_indicators_dict[source]['pearson_d2'] = 0.0
                additional_indicators_dict[source]['cosine_num'] = 0.0
                additional_indicators_dict[source]['cosine_den1'] = 0.0
                additional_indicators_dict[source]['cosine_den2'] = 0.0
            for t in range(1, len(data_df['station_q'])):
                dt = data_df['time_in_days'].iloc[t] - data_df['time_in_days'].iloc[t - 1]
                sq = data_df['station_q'].iloc
                for source in additional_indicators_dict.keys():
                    q = additional_indicators_dict[source]['array_value']
                    bias = additional_indicators_dict[source]['bias']
                    additional_indicators_dict[source]['nrmse_sum'] += ((q[t] - sq[t]) ** 2 + (q[t - 1] - sq[t - 1]) ** 2) / 2 * dt
                    additional_indicators_dict[source]['nrmse2_sum'] += ((q[t] - sq[t] - bias) ** 2 + (q[t - 1] - sq[t - 1] - bias) ** 2) / 2 * dt
                    p_nom, p_d1, p_d2 = interpolated_pearson(q, additional_indicators_dict[source]['mean_value'], sq, Q_ref_mean, t, data_df['time_in_days'].iloc)
                    additional_indicators_dict[source]['pearson_nom'] += p_nom
                    additional_indicators_dict[source]['pearson_d1'] += p_d1
                    additional_indicators_dict[source]['pearson_d2'] += p_d2
                    c_nom, c_d1, c_d2 = interpolated_pearson(q, 0.0, sq, 0.0, t, data_df['time_in_days'].iloc)
                    additional_indicators_dict[source]['cosine_num'] += c_nom
                    additional_indicators_dict[source]['cosine_den1'] += c_d1
                    additional_indicators_dict[source]['cosine_den2'] += c_d2
                time_scaling += dt
            for source in additional_indicators_dict.keys():
                additional_indicators_dict[source]['nrmse'] = np.sqrt(additional_indicators_dict[source]['nrmse_sum'] / time_scaling) / Q_ref_mean
                additional_indicators_dict[source]['nrmse2'] = np.sqrt(additional_indicators_dict[source]['nrmse2_sum'] / time_scaling) / Q_ref_mean
                if additional_indicators_dict[source]['pearson_d1'] > 0.0 and additional_indicators_dict[source]['pearson_d2'] > 0.0:
                    additional_indicators_dict[source]['pearson'] = additional_indicators_dict[source]['pearson_nom'] / np.sqrt(additional_indicators_dict[source]['pearson_d1'] * additional_indicators_dict[source]['pearson_d2'])
                else:
                    additional_indicators_dict[source]['pearson'] = -9999.0
                additional_indicators_dict[source]['cosine_similarity'] = additional_indicators_dict[source]['cosine_num'] / np.sqrt(additional_indicators_dict[source]['cosine_den1'] * additional_indicators_dict[source]['cosine_den2'])
                q_col = np.array(additional_indicators_dict[source]['array_value'])
                bias = additional_indicators_dict[source]['bias'] / Q_ref_mean
                additional_indicators_dict[source]['log_cosh'] = log_cosh_loss(data_df['station_q'] / Q_ref_mean, q_col / Q_ref_mean, bias=bias)
                additional_indicators_dict[source]['tweedie_gamma'] = np.sum(tweedie_loss(data_df['station_q'], q_col, power=2.0))
                if additional_indicators_dict[source]['pearson'] != -9999.0:
                    additional_indicators_dict[source]['KGE_nrmse'] = 1 - np.sqrt((1 - additional_indicators_dict[source]['pearson']) ** 2 + bias ** 2 + additional_indicators_dict[source]['nrmse2'] ** 2)
                    additional_indicators_dict[source]['KGE'] = 1 - np.sqrt((1 - additional_indicators_dict[source]['pearson']) ** 2 + bias ** 2 + (additional_indicators_dict[source]['var_value'] / Q_ref_var - 1) ** 2)
                    additional_indicators_dict[source]['KGE_cosh'] = 1 - np.sqrt((1 - additional_indicators_dict[source]['pearson']) ** 2 + bias ** 2 + additional_indicators_dict[source]['log_cosh'] ** 2)
                else:
                    additional_indicators_dict[source]['KGE_nrmse'] = 1 - np.sqrt((1 - 0.0) ** 2 + bias ** 2 + additional_indicators_dict[source]['nrmse2'] ** 2)
                    additional_indicators_dict[source]['KGE'] = 1 - np.sqrt((1 - 0.0) ** 2 + bias ** 2 + (additional_indicators_dict[source]['var_value'] / Q_ref_var - 1) ** 2)
                    additional_indicators_dict[source]['KGE_cosh'] = 1 - np.sqrt((1 - 0.0) ** 2 + bias ** 2 + additional_indicators_dict[source]['log_cosh'] ** 2)
                additional_indicators_dict[source]['spearman'] = spearman_correlation(data_df['station_q'], q_col)[0]
        elif not indicators_experimental_computation:
            data_df = sic4dvar_df.merge(station_df, how='inner', left_on='sic4dvar_date', right_on='station_date')
        if len(data_df) >= min_dates:
            indicators = {'reach_id': str(reach_id), 'nb_dates': len(data_df)}
            if not lite_run:
                indicators['relative_error'] = (additional_indicators_dict['algo']['cv'] - additional_indicators_dict['truth']['cv']) / additional_indicators_dict['truth']['cv'] if additional_indicators_dict['truth']['cv'] != 0 else np.nan
            if indicators_experimental_computation:
                if np.array(x).size > 0:
                    indicators['x0'] = x[0]
                    indicators['x1'] = x[1]
                for key, value in additional_indicators_dict.items():
                    for indicator in indicators_list:
                        if indicator in value.keys():
                            indicators[f'{key}_{indicator}'] = value[indicator]
                    indicators[f'{key}_relative_error'] = indicators['relative_error']
                indicators['data_df'] = data_df
            else:
                pass
                logging.info('Indicators experimental computation not enabled, skipping indicators computation.')
            logging.info(f'Indicators name : {';'.join(list(indicators.keys()))}')
            logging.info(f'Indicators values : {';'.join([str(e) for e in indicators.values()])}')
            if lite_run:
                indicators.pop('data_df', None)
                indicators.pop('reach_id', None)
            return indicators
        else:
            logging.info(f"Less than {min_dates} valid dates for true/prediction comparison. Can't compute indicators.")
            return {}
    else:
        logging.info("No station data. Can't compute indicators.")
        return {}

def main():
    y_true = np.array([2, 3, 4, 5, 6])
    y_pred = np.array([4, 1, 3, 2, 0])
    time_frequency_test = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    indic = compute_all_indicators_from_predict_true(y_true, y_pred)
    indic_time = compute_all_indicators_from_predict_true(y_true, y_pred, time_frequency_test)
    print('indicator value_without_time value_with_time')
    for k in indic.keys():
        print(k, indic[k], indic_time[k])
    algo_q = '[27.982093447684708 11.872623186343 8.676329345082493 16.808082292981574\n    14.154325113753528 6.798925800952132 -- 12.008800637302041\n    147.5788055797178 -- 21.258396505422777 9.707385162909862\n    5.362994808978341 22.398074780235014 13.779968585251332 4.654902742135091\n    -- -- -- -- -- 22.028501691249257 7.545237454688801 -- 36.12185142263048\n    37.435334661603996 20.694450608415174 8.014338005446916 18.42876520298548\n    7.8487082669164945 4.893847284715418 30.716153920123556\n    11.727921031281241 4.200489494164043 11.428661955258121\n    2.8866602456053574 4.058978642759522 4.194298337113741 18.525855493531527\n    7.379783211798431 -- -- -- -- -- --]'
    algo_t = '[8612.295930088978 8664.407612004628 8674.889841980645 8685.2722226171\n    8695.75447702877 8706.136868375319 -- 8727.001512282362 8737.48376773385\n    -- 8768.730799829933 8779.213046217403 8800.077698949483\n    8810.460084686762 8820.94234245859 8831.324717507005 -- -- -- -- --\n    8904.400873982739 8914.783260178345 -- 8935.647903512609\n    8946.130142165486 8956.512519504244 8966.994783248972 8998.241795931146\n    9019.106449201861 9029.588716881459 9039.971125186563 9050.453357608267\n    9060.835752836345 9113.047267440767 9144.294276583834 9154.776539586677\n    9165.158913705827 9175.641164617284 9186.023563600444 -- -- -- -- -- --]'
    station_q = '[33.06 29.044 27.751 26.237 48.323 27.732 30.487 28.631 25.277 22.29 19.355 17.213 16.596 15.024 14.16 13.631\n    15.074 14.112 15.276 16.701 16.503 15.739 16.21 16.899 17.55 14.952 14.223 13.697 13.087 27.345 37.515 27.647\n    44.804 33.674 23.931 19.113 16.73 14.677 13.431 13.84 18.994 16.512 14.616 20.567 29.329 26.013 18.321 16.282\n    15.009 13.623 10.891 10.208 23.624 15.422 12.806 12.366 11.491 11.081 10.838 9.695 8.894 8.743 8.292 8.042\n    16.854 15.843 11.157 8.778 8.351 8.426 8.525 8.556 8.625 8.044 8.555 7.559 6.734 6.661 7.821 7.091\n    6.869 8.373 13.32 12.162 20.635 17.528 16.731 15.237 24.03 19.409 16.868 17.194 16.399 14.573 13.201 12.379\n    15.85 39.033 22.091 16.583 15.642 14.146 13.416 12.99 14.915 15.893 17.189 14.439 14.378 13.411 12.565 12.404\n    12.555 10.72 9.977 9.762 9.465 8.951 16.181 69.282 78.292 38.99 46.617 38.045 29.706 23.585 25.075 43.33\n    32.735 27.875 28.968 26.945 23.644 20.906 19.154 17.792 16.385 15.376 15.289 14.401 14.073 13.225 13.309 12.947\n    12.664 12.602 12.675 12.46 12.93 12.859 12.434 12.281 13.693 12.725 12.804 12.738 12.411 12.874 14.568 20.819\n    16.84 14.725 13.42 13.452 14.875 14.468 14.771 14.911 14.662 14.545 16.127 15.135 13.458 17.764 23.71 20.458\n    19.642 21.231 20.405 22.284 21.585 19.371 18.119 17.297 16.453 16.851 17.009 16.396 15.818 19.519 19.284 18.312\n    18.533 18.835 29.597 24.95 21.535 19.56 17.533 16.278 15.86 16.124 14.676 14.416 14.407 13.836 13.588 12.958\n    12.415 12.174 12.984 15.112 14.915 14.53 14.178 14.612 16.84 17.62 16.769 16.412 16.121 14.684 13.271 12.829\n    12.489 13.535 17.038 16.08 14.98 23.663 22.767 19.612 17.734 16.793 18.009 17.049 14.409 13.575 13.684 13.092\n    12.429 11.941 11.723 12.337 12.765 12.7 12.621 12.446 12.223 12.009 18.585 19.734 17.375 15.892 15.63 14.925\n    15.634 16.141 15.403 14.548 13.757 13.042 12.538 11.366 11.768 17.248 24.744 19.093 14.831 13.393 12.911 12.247\n    10.829 9.906 9.164 8.579 9.133 24.295 21.9 18.678 20.027 19.481 19.399 17.747 15.384 14.922 13.688 12.144\n    11.668 12.289 12.721 11.561 10.218 9.573 9.314 8.878 8.103 8.25 8.085 7.727 7.455 7.333 7.182 7.333\n    7.59 7.449 6.707 6.161 6.31 6.089 5.881 5.753 6.499 9.848 10.975 10.925 9.588 8.368 8.196 7.663\n    6.862 6.164 5.764 5.574 17.535 12.924 11.619 11.489 11.752 17.443 13.484 11.088 93.38 159.398 143.472 160.102\n    150.261 156.488 97.238 66.576 45.773 34.876 30.872 26.731 23.594 22.149 19.97 16.989 15.513 13.779 11.577 10.359\n    10.278 18.111 21.124 17.038 18.23 32.601 34.338 37.81 49.347 100.567 111.26 76.366 48.547 39.221 31.476 42.869\n    55.706 96.618 96.403 69.551 77.649 91.197 87.404 64.264 47.379 46.719 107.776 181.961 112.048 83.731 106.799 84.946\n    68.722 53.02 39.836 37.068 31.241 27.533 25.097 22.331 21.358 19.969 17.831 16.992 16.129 14.783 13.787 12.919\n    12.874 26.045 17.82 15.166 14.373 13.122 13.56 12.162 11.352 15.896 13.828 12.035 11.435 15.31 17.105 17.484\n    16.579 17.584 14.017 13.085 11.843 10.322 9.323 8.594 8.101 7.691 7.625 7.565 6.769 6.108 5.734 5.596\n    5.36 5.706 5.453 5.91 5.965 5.149 5.205 7.704 8.981 7.162 8.029 8.806 11.068 15.95 10.964 8.839\n    9.079 9.404 8.713 7.93 7.426 6.945 6.709 6.468 6.101 6.011 5.819 5.725 7.377 74.03 57.315 30.791\n    54.193 44.214 30.886 26.891 21.02 18.7 16.042 14.168 13.059 12.149 11.602 10.412 8.876 9.427 10.033 9.769\n    9.246 9.088 8.424 8.334 8.491 8.627 8.937 8.98 9.31 8.774 8.743 9.263 8.586 7.432 8.338 12.942\n    14.509 22.315 24.058 53.598 47.045 35.798 27.469 24.939 21.04 18.342 16.599 17.098 15.805 14.411 13.709 12.171\n    10.534 9.45 9.64 12.053 13.067 11.162 12.688 13.311 12.449 12.037 14.507 19.321 27.351 28.257 56.854 48.084\n    36.587 28.472 25.336 23.462 23.665 22.729 24.453 22.572 20.484 19.018 18.29 17.392 16.451 14.622 15.308 15.243\n    14.865 14.545 14.409 15.643 14.718 14.161 13.436 13.597 13.093 12.53 12.681 15.913 18.952 14.716 13.342 14.937\n    15.185 14.735 14.174 12.885 12.554 13.639 16.445 24.575 21.375 17.599 19.439 20.635 27.742 27.214 22.456 18.958\n    18.366 15.909 15.476 14.57 13.852 13.733 13.77 13.512 13.25 12.842 12.817 12.843 13.523 14.946 17.118 16.347\n    14.336 12.444 13.446 12.921 12.84 12.663 12.317 12.674 15.802 24.963 26.313 24.641 24.37 21.85 22.92 20.979\n    18.246 15.67 14.962 12.84 12.132 12.2 12.989 12.337 11.769 11.144 10.87 10.559 10.914 9.743 9.12 8.957\n    8.575 8.675 8.748 8.779 9.519 9.884 9.198 9.415 9.239 10.385 14.013 15.93 22.75 26.328 22.257 18.345\n    15.942 13.277 12.083 12.172 13.118 11.083 10.2 9.357 9.577 9.47 9.133 8.842 8.674 10.665 11.284 10.595\n    11.12 13.949 13.188 11.333 9.802 8.963 8.601 8.105 7.363 7.193 7.651 10.422 9.496 8.706 7.736 6.957\n    6.622 6.576 5.98 5.352 5.007 4.688 4.436 4.587 5.228 5.24 4.671 4.334 5.859 5.36 6.206 14.439\n    12.133 10.355 8.55 6.415 5.561 5.738 5.178 4.421 4.318 4.324 4.33 38.715 104.4 45.831 25.023 20.687\n    21.848 16.208 52.956 35.667 21.844 14.062 10.764 10.294 9.712 8.116 6.732 5.776 5.16 4.699 4.124 3.86\n    3.571 3.335 3.156 3.256 2.903 2.594 2.438 2.432 2.281 2.266 2.297 2.21 2.118 2.085 2.048 1.959\n    1.937 2.074 1.955 1.909 1.967 2.067 37.421 16.529 15.753 12.351 27.783 15.916 12.256 9.088 7.406 8.673\n    7.596 6.255 5.458 4.8 4.658 4.225 4.065 3.638 3.349 3.25 4.565 5.143 4.9 3.957 3.111 3.755\n    4.517 4.449 4.486 7.093 24.666 21.255 18.583 17.971 25.956 15.523 18.963 13.436 10.939 9.119 7.691 6.533\n    6.025 5.628 4.932 4.36 4.004 3.675 3.421 3.251 3.048 2.782 2.565 2.399 2.308 2.196 2.21 2.338\n    2.924 6.351 5.101 4.067 3.63 4.134 3.541 3.17 2.971 2.833 2.879 3.502 3.982 4.016 3.495 3.174\n    2.891 2.624 4.1 15.744 9.234 5.95 5.023 5.681 8.038 10.74 10.47 7.889 6.126 5.425 4.923 4.449\n    4.152 3.774 3.742 3.883 4.153 4.562 3.978 3.323 3.213 3.829 4.289 6.251 5.34 5.474 4.558 12.227\n    14.909 11.724 13.725 13.904 10.9 9.069 8.088 7.841 7.503 6.699 6.228 5.74 5.11 -- -- --\n    -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --\n    -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --\n    -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --\n    -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --\n    -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --\n    -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --\n    -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --\n    -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --\n    -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --\n    -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --\n    -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --\n    -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --\n    -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --\n    -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --\n    -- -- -- -- -- -- -- --]'
    station_date = [datetime.datetime(2023, 1, 1, 0, 0), datetime.datetime(2023, 1, 2, 0, 0), datetime.datetime(2023, 1, 3, 0, 0), datetime.datetime(2023, 1, 4, 0, 0), datetime.datetime(2023, 1, 5, 0, 0), datetime.datetime(2023, 1, 6, 0, 0), datetime.datetime(2023, 1, 7, 0, 0), datetime.datetime(2023, 1, 8, 0, 0), datetime.datetime(2023, 1, 9, 0, 0), datetime.datetime(2023, 1, 10, 0, 0), datetime.datetime(2023, 1, 11, 0, 0), datetime.datetime(2023, 1, 12, 0, 0), datetime.datetime(2023, 1, 13, 0, 0), datetime.datetime(2023, 1, 14, 0, 0), datetime.datetime(2023, 1, 15, 0, 0), datetime.datetime(2023, 1, 16, 0, 0), datetime.datetime(2023, 1, 17, 0, 0), datetime.datetime(2023, 1, 18, 0, 0), datetime.datetime(2023, 1, 19, 0, 0), datetime.datetime(2023, 1, 20, 0, 0), datetime.datetime(2023, 1, 21, 0, 0), datetime.datetime(2023, 1, 22, 0, 0), datetime.datetime(2023, 1, 23, 0, 0), datetime.datetime(2023, 1, 24, 0, 0), datetime.datetime(2023, 1, 25, 0, 0), datetime.datetime(2023, 1, 26, 0, 0), datetime.datetime(2023, 1, 27, 0, 0), datetime.datetime(2023, 1, 28, 0, 0), datetime.datetime(2023, 1, 29, 0, 0), datetime.datetime(2023, 1, 30, 0, 0), datetime.datetime(2023, 1, 31, 0, 0), datetime.datetime(2023, 2, 1, 0, 0), datetime.datetime(2023, 2, 2, 0, 0), datetime.datetime(2023, 2, 3, 0, 0), datetime.datetime(2023, 2, 4, 0, 0), datetime.datetime(2023, 2, 5, 0, 0), datetime.datetime(2023, 2, 6, 0, 0), datetime.datetime(2023, 2, 7, 0, 0), datetime.datetime(2023, 2, 8, 0, 0), datetime.datetime(2023, 2, 9, 0, 0), datetime.datetime(2023, 2, 10, 0, 0), datetime.datetime(2023, 2, 11, 0, 0), datetime.datetime(2023, 2, 12, 0, 0), datetime.datetime(2023, 2, 13, 0, 0), datetime.datetime(2023, 2, 14, 0, 0), datetime.datetime(2023, 2, 15, 0, 0), datetime.datetime(2023, 2, 16, 0, 0), datetime.datetime(2023, 2, 17, 0, 0), datetime.datetime(2023, 2, 18, 0, 0), datetime.datetime(2023, 2, 19, 0, 0), datetime.datetime(2023, 2, 20, 0, 0), datetime.datetime(2023, 2, 21, 0, 0), datetime.datetime(2023, 2, 22, 0, 0), datetime.datetime(2023, 2, 23, 0, 0), datetime.datetime(2023, 2, 24, 0, 0), datetime.datetime(2023, 2, 25, 0, 0), datetime.datetime(2023, 2, 26, 0, 0), datetime.datetime(2023, 2, 27, 0, 0), datetime.datetime(2023, 2, 28, 0, 0), datetime.datetime(2023, 3, 1, 0, 0), datetime.datetime(2023, 3, 2, 0, 0), datetime.datetime(2023, 3, 3, 0, 0), datetime.datetime(2023, 3, 4, 0, 0), datetime.datetime(2023, 3, 5, 0, 0), datetime.datetime(2023, 3, 6, 0, 0), datetime.datetime(2023, 3, 7, 0, 0), datetime.datetime(2023, 3, 8, 0, 0), datetime.datetime(2023, 3, 9, 0, 0), datetime.datetime(2023, 3, 10, 0, 0), datetime.datetime(2023, 3, 11, 0, 0), datetime.datetime(2023, 3, 12, 0, 0), datetime.datetime(2023, 3, 13, 0, 0), datetime.datetime(2023, 3, 14, 0, 0), datetime.datetime(2023, 3, 15, 0, 0), datetime.datetime(2023, 3, 16, 0, 0), datetime.datetime(2023, 3, 17, 0, 0), datetime.datetime(2023, 3, 18, 0, 0), datetime.datetime(2023, 3, 19, 0, 0), datetime.datetime(2023, 3, 20, 0, 0), datetime.datetime(2023, 3, 21, 0, 0), datetime.datetime(2023, 3, 22, 0, 0), datetime.datetime(2023, 3, 23, 0, 0), datetime.datetime(2023, 3, 24, 0, 0), datetime.datetime(2023, 3, 25, 0, 0), datetime.datetime(2023, 3, 26, 0, 0), datetime.datetime(2023, 3, 27, 0, 0), datetime.datetime(2023, 3, 28, 0, 0), datetime.datetime(2023, 3, 29, 0, 0), datetime.datetime(2023, 3, 30, 0, 0), datetime.datetime(2023, 3, 31, 0, 0), datetime.datetime(2023, 4, 1, 0, 0), datetime.datetime(2023, 4, 2, 0, 0), datetime.datetime(2023, 4, 3, 0, 0), datetime.datetime(2023, 4, 4, 0, 0), datetime.datetime(2023, 4, 5, 0, 0), datetime.datetime(2023, 4, 6, 0, 0), datetime.datetime(2023, 4, 7, 0, 0), datetime.datetime(2023, 4, 8, 0, 0), datetime.datetime(2023, 4, 9, 0, 0), datetime.datetime(2023, 4, 10, 0, 0), datetime.datetime(2023, 4, 11, 0, 0), datetime.datetime(2023, 4, 12, 0, 0), datetime.datetime(2023, 4, 13, 0, 0), datetime.datetime(2023, 4, 14, 0, 0), datetime.datetime(2023, 4, 15, 0, 0), datetime.datetime(2023, 4, 16, 0, 0), datetime.datetime(2023, 4, 17, 0, 0), datetime.datetime(2023, 4, 18, 0, 0), datetime.datetime(2023, 4, 19, 0, 0), datetime.datetime(2023, 4, 20, 0, 0), datetime.datetime(2023, 4, 21, 0, 0), datetime.datetime(2023, 4, 22, 0, 0), datetime.datetime(2023, 4, 23, 0, 0), datetime.datetime(2023, 4, 24, 0, 0), datetime.datetime(2023, 4, 25, 0, 0), datetime.datetime(2023, 4, 26, 0, 0), datetime.datetime(2023, 4, 27, 0, 0), datetime.datetime(2023, 4, 28, 0, 0), datetime.datetime(2023, 4, 29, 0, 0), datetime.datetime(2023, 4, 30, 0, 0), datetime.datetime(2023, 5, 1, 0, 0), datetime.datetime(2023, 5, 2, 0, 0), datetime.datetime(2023, 5, 3, 0, 0), datetime.datetime(2023, 5, 4, 0, 0), datetime.datetime(2023, 5, 5, 0, 0), datetime.datetime(2023, 5, 6, 0, 0), datetime.datetime(2023, 5, 7, 0, 0), datetime.datetime(2023, 5, 8, 0, 0), datetime.datetime(2023, 5, 9, 0, 0), datetime.datetime(2023, 5, 10, 0, 0), datetime.datetime(2023, 5, 11, 0, 0), datetime.datetime(2023, 5, 12, 0, 0), datetime.datetime(2023, 5, 13, 0, 0), datetime.datetime(2023, 5, 14, 0, 0), datetime.datetime(2023, 5, 15, 0, 0), datetime.datetime(2023, 5, 16, 0, 0), datetime.datetime(2023, 5, 17, 0, 0), datetime.datetime(2023, 5, 18, 0, 0), datetime.datetime(2023, 5, 19, 0, 0), datetime.datetime(2023, 5, 20, 0, 0), datetime.datetime(2023, 5, 21, 0, 0), datetime.datetime(2023, 5, 22, 0, 0), datetime.datetime(2023, 5, 23, 0, 0), datetime.datetime(2023, 5, 24, 0, 0), datetime.datetime(2023, 5, 25, 0, 0), datetime.datetime(2023, 5, 26, 0, 0), datetime.datetime(2023, 5, 27, 0, 0), datetime.datetime(2023, 5, 28, 0, 0), datetime.datetime(2023, 5, 29, 0, 0), datetime.datetime(2023, 5, 30, 0, 0), datetime.datetime(2023, 5, 31, 0, 0), datetime.datetime(2023, 6, 1, 0, 0), datetime.datetime(2023, 6, 2, 0, 0), datetime.datetime(2023, 6, 3, 0, 0), datetime.datetime(2023, 6, 4, 0, 0), datetime.datetime(2023, 6, 5, 0, 0), datetime.datetime(2023, 6, 6, 0, 0), datetime.datetime(2023, 6, 7, 0, 0), datetime.datetime(2023, 6, 8, 0, 0), datetime.datetime(2023, 6, 9, 0, 0), datetime.datetime(2023, 6, 10, 0, 0), datetime.datetime(2023, 6, 11, 0, 0), datetime.datetime(2023, 6, 12, 0, 0), datetime.datetime(2023, 6, 13, 0, 0), datetime.datetime(2023, 6, 14, 0, 0), datetime.datetime(2023, 6, 15, 0, 0), datetime.datetime(2023, 6, 16, 0, 0), datetime.datetime(2023, 6, 17, 0, 0), datetime.datetime(2023, 6, 18, 0, 0), datetime.datetime(2023, 6, 19, 0, 0), datetime.datetime(2023, 6, 20, 0, 0), datetime.datetime(2023, 6, 21, 0, 0), datetime.datetime(2023, 6, 22, 0, 0), datetime.datetime(2023, 6, 23, 0, 0), datetime.datetime(2023, 6, 24, 0, 0), datetime.datetime(2023, 6, 25, 0, 0), datetime.datetime(2023, 6, 26, 0, 0), datetime.datetime(2023, 6, 27, 0, 0), datetime.datetime(2023, 6, 28, 0, 0), datetime.datetime(2023, 6, 29, 0, 0), datetime.datetime(2023, 6, 30, 0, 0), datetime.datetime(2023, 7, 1, 0, 0), datetime.datetime(2023, 7, 2, 0, 0), datetime.datetime(2023, 7, 3, 0, 0), datetime.datetime(2023, 7, 4, 0, 0), datetime.datetime(2023, 7, 5, 0, 0), datetime.datetime(2023, 7, 6, 0, 0), datetime.datetime(2023, 7, 7, 0, 0), datetime.datetime(2023, 7, 8, 0, 0), datetime.datetime(2023, 7, 9, 0, 0), datetime.datetime(2023, 7, 10, 0, 0), datetime.datetime(2023, 7, 11, 0, 0), datetime.datetime(2023, 7, 12, 0, 0), datetime.datetime(2023, 7, 13, 0, 0), datetime.datetime(2023, 7, 14, 0, 0), datetime.datetime(2023, 7, 15, 0, 0), datetime.datetime(2023, 7, 16, 0, 0), datetime.datetime(2023, 7, 17, 0, 0), datetime.datetime(2023, 7, 18, 0, 0), datetime.datetime(2023, 7, 19, 0, 0), datetime.datetime(2023, 7, 20, 0, 0), datetime.datetime(2023, 7, 21, 0, 0), datetime.datetime(2023, 7, 22, 0, 0), datetime.datetime(2023, 7, 23, 0, 0), datetime.datetime(2023, 7, 24, 0, 0), datetime.datetime(2023, 7, 25, 0, 0), datetime.datetime(2023, 7, 26, 0, 0), datetime.datetime(2023, 7, 27, 0, 0), datetime.datetime(2023, 7, 28, 0, 0), datetime.datetime(2023, 7, 29, 0, 0), datetime.datetime(2023, 7, 30, 0, 0), datetime.datetime(2023, 7, 31, 0, 0), datetime.datetime(2023, 8, 1, 0, 0), datetime.datetime(2023, 8, 2, 0, 0), datetime.datetime(2023, 8, 3, 0, 0), datetime.datetime(2023, 8, 4, 0, 0), datetime.datetime(2023, 8, 5, 0, 0), datetime.datetime(2023, 8, 6, 0, 0), datetime.datetime(2023, 8, 7, 0, 0), datetime.datetime(2023, 8, 8, 0, 0), datetime.datetime(2023, 8, 9, 0, 0), datetime.datetime(2023, 8, 10, 0, 0), datetime.datetime(2023, 8, 11, 0, 0), datetime.datetime(2023, 8, 12, 0, 0), datetime.datetime(2023, 8, 13, 0, 0), datetime.datetime(2023, 8, 14, 0, 0), datetime.datetime(2023, 8, 15, 0, 0), datetime.datetime(2023, 8, 16, 0, 0), datetime.datetime(2023, 8, 17, 0, 0), datetime.datetime(2023, 8, 18, 0, 0), datetime.datetime(2023, 8, 19, 0, 0), datetime.datetime(2023, 8, 20, 0, 0), datetime.datetime(2023, 8, 21, 0, 0), datetime.datetime(2023, 8, 22, 0, 0), datetime.datetime(2023, 8, 23, 0, 0), datetime.datetime(2023, 8, 24, 0, 0), datetime.datetime(2023, 8, 25, 0, 0), datetime.datetime(2023, 8, 26, 0, 0), datetime.datetime(2023, 8, 27, 0, 0), datetime.datetime(2023, 8, 28, 0, 0), datetime.datetime(2023, 8, 29, 0, 0), datetime.datetime(2023, 8, 30, 0, 0), datetime.datetime(2023, 8, 31, 0, 0), datetime.datetime(2023, 9, 1, 0, 0), datetime.datetime(2023, 9, 2, 0, 0), datetime.datetime(2023, 9, 3, 0, 0), datetime.datetime(2023, 9, 4, 0, 0), datetime.datetime(2023, 9, 5, 0, 0), datetime.datetime(2023, 9, 6, 0, 0), datetime.datetime(2023, 9, 7, 0, 0), datetime.datetime(2023, 9, 8, 0, 0), datetime.datetime(2023, 9, 9, 0, 0), datetime.datetime(2023, 9, 10, 0, 0), datetime.datetime(2023, 9, 11, 0, 0), datetime.datetime(2023, 9, 12, 0, 0), datetime.datetime(2023, 9, 13, 0, 0), datetime.datetime(2023, 9, 14, 0, 0), datetime.datetime(2023, 9, 15, 0, 0), datetime.datetime(2023, 9, 16, 0, 0), datetime.datetime(2023, 9, 17, 0, 0), datetime.datetime(2023, 9, 18, 0, 0), datetime.datetime(2023, 9, 19, 0, 0), datetime.datetime(2023, 9, 20, 0, 0), datetime.datetime(2023, 9, 21, 0, 0), datetime.datetime(2023, 9, 22, 0, 0), datetime.datetime(2023, 9, 23, 0, 0), datetime.datetime(2023, 9, 24, 0, 0), datetime.datetime(2023, 9, 25, 0, 0), datetime.datetime(2023, 9, 26, 0, 0), datetime.datetime(2023, 9, 27, 0, 0), datetime.datetime(2023, 9, 28, 0, 0), datetime.datetime(2023, 9, 29, 0, 0), datetime.datetime(2023, 9, 30, 0, 0), datetime.datetime(2023, 10, 1, 0, 0), datetime.datetime(2023, 10, 2, 0, 0), datetime.datetime(2023, 10, 3, 0, 0), datetime.datetime(2023, 10, 4, 0, 0), datetime.datetime(2023, 10, 5, 0, 0), datetime.datetime(2023, 10, 6, 0, 0), datetime.datetime(2023, 10, 7, 0, 0), datetime.datetime(2023, 10, 8, 0, 0), datetime.datetime(2023, 10, 9, 0, 0), datetime.datetime(2023, 10, 10, 0, 0), datetime.datetime(2023, 10, 11, 0, 0), datetime.datetime(2023, 10, 12, 0, 0), datetime.datetime(2023, 10, 13, 0, 0), datetime.datetime(2023, 10, 14, 0, 0), datetime.datetime(2023, 10, 15, 0, 0), datetime.datetime(2023, 10, 16, 0, 0), datetime.datetime(2023, 10, 17, 0, 0), datetime.datetime(2023, 10, 18, 0, 0), datetime.datetime(2023, 10, 19, 0, 0), datetime.datetime(2023, 10, 20, 0, 0), datetime.datetime(2023, 10, 21, 0, 0), datetime.datetime(2023, 10, 22, 0, 0), datetime.datetime(2023, 10, 23, 0, 0), datetime.datetime(2023, 10, 24, 0, 0), datetime.datetime(2023, 10, 25, 0, 0), datetime.datetime(2023, 10, 26, 0, 0), datetime.datetime(2023, 10, 27, 0, 0), datetime.datetime(2023, 10, 28, 0, 0), datetime.datetime(2023, 10, 29, 0, 0), datetime.datetime(2023, 10, 30, 0, 0), datetime.datetime(2023, 10, 31, 0, 0), datetime.datetime(2023, 11, 1, 0, 0), datetime.datetime(2023, 11, 2, 0, 0), datetime.datetime(2023, 11, 3, 0, 0), datetime.datetime(2023, 11, 4, 0, 0), datetime.datetime(2023, 11, 5, 0, 0), datetime.datetime(2023, 11, 6, 0, 0), datetime.datetime(2023, 11, 7, 0, 0), datetime.datetime(2023, 11, 8, 0, 0), datetime.datetime(2023, 11, 9, 0, 0), datetime.datetime(2023, 11, 10, 0, 0), datetime.datetime(2023, 11, 11, 0, 0), datetime.datetime(2023, 11, 12, 0, 0), datetime.datetime(2023, 11, 13, 0, 0), datetime.datetime(2023, 11, 14, 0, 0), datetime.datetime(2023, 11, 15, 0, 0), datetime.datetime(2023, 11, 16, 0, 0), datetime.datetime(2023, 11, 17, 0, 0), datetime.datetime(2023, 11, 18, 0, 0), datetime.datetime(2023, 11, 19, 0, 0), datetime.datetime(2023, 11, 20, 0, 0), datetime.datetime(2023, 11, 21, 0, 0), datetime.datetime(2023, 11, 22, 0, 0), datetime.datetime(2023, 11, 23, 0, 0), datetime.datetime(2023, 11, 24, 0, 0), datetime.datetime(2023, 11, 25, 0, 0), datetime.datetime(2023, 11, 26, 0, 0), datetime.datetime(2023, 11, 27, 0, 0), datetime.datetime(2023, 11, 28, 0, 0), datetime.datetime(2023, 11, 29, 0, 0), datetime.datetime(2023, 11, 30, 0, 0), datetime.datetime(2023, 12, 1, 0, 0), datetime.datetime(2023, 12, 2, 0, 0), datetime.datetime(2023, 12, 3, 0, 0), datetime.datetime(2023, 12, 4, 0, 0), datetime.datetime(2023, 12, 5, 0, 0), datetime.datetime(2023, 12, 6, 0, 0), datetime.datetime(2023, 12, 7, 0, 0), datetime.datetime(2023, 12, 8, 0, 0), datetime.datetime(2023, 12, 9, 0, 0), datetime.datetime(2023, 12, 10, 0, 0), datetime.datetime(2023, 12, 11, 0, 0), datetime.datetime(2023, 12, 12, 0, 0), datetime.datetime(2023, 12, 13, 0, 0), datetime.datetime(2023, 12, 14, 0, 0), datetime.datetime(2023, 12, 15, 0, 0), datetime.datetime(2023, 12, 16, 0, 0), datetime.datetime(2023, 12, 17, 0, 0), datetime.datetime(2023, 12, 18, 0, 0), datetime.datetime(2023, 12, 19, 0, 0), datetime.datetime(2023, 12, 20, 0, 0), datetime.datetime(2023, 12, 21, 0, 0), datetime.datetime(2023, 12, 22, 0, 0), datetime.datetime(2023, 12, 23, 0, 0), datetime.datetime(2023, 12, 24, 0, 0), datetime.datetime(2023, 12, 25, 0, 0), datetime.datetime(2023, 12, 26, 0, 0), datetime.datetime(2023, 12, 27, 0, 0), datetime.datetime(2023, 12, 28, 0, 0), datetime.datetime(2023, 12, 29, 0, 0), datetime.datetime(2023, 12, 30, 0, 0), datetime.datetime(2023, 12, 31, 0, 0), datetime.datetime(2024, 1, 1, 0, 0), datetime.datetime(2024, 1, 2, 0, 0), datetime.datetime(2024, 1, 3, 0, 0), datetime.datetime(2024, 1, 4, 0, 0), datetime.datetime(2024, 1, 5, 0, 0), datetime.datetime(2024, 1, 6, 0, 0), datetime.datetime(2024, 1, 7, 0, 0), datetime.datetime(2024, 1, 8, 0, 0), datetime.datetime(2024, 1, 9, 0, 0), datetime.datetime(2024, 1, 10, 0, 0), datetime.datetime(2024, 1, 11, 0, 0), datetime.datetime(2024, 1, 12, 0, 0), datetime.datetime(2024, 1, 13, 0, 0), datetime.datetime(2024, 1, 14, 0, 0), datetime.datetime(2024, 1, 15, 0, 0), datetime.datetime(2024, 1, 16, 0, 0), datetime.datetime(2024, 1, 17, 0, 0), datetime.datetime(2024, 1, 18, 0, 0), datetime.datetime(2024, 1, 19, 0, 0), datetime.datetime(2024, 1, 20, 0, 0), datetime.datetime(2024, 1, 21, 0, 0), datetime.datetime(2024, 1, 22, 0, 0), datetime.datetime(2024, 1, 23, 0, 0), datetime.datetime(2024, 1, 24, 0, 0), datetime.datetime(2024, 1, 25, 0, 0), datetime.datetime(2024, 1, 26, 0, 0), datetime.datetime(2024, 1, 27, 0, 0), datetime.datetime(2024, 1, 28, 0, 0), datetime.datetime(2024, 1, 29, 0, 0), datetime.datetime(2024, 1, 30, 0, 0), datetime.datetime(2024, 1, 31, 0, 0), datetime.datetime(2024, 2, 1, 0, 0), datetime.datetime(2024, 2, 2, 0, 0), datetime.datetime(2024, 2, 3, 0, 0), datetime.datetime(2024, 2, 4, 0, 0), datetime.datetime(2024, 2, 5, 0, 0), datetime.datetime(2024, 2, 6, 0, 0), datetime.datetime(2024, 2, 7, 0, 0), datetime.datetime(2024, 2, 8, 0, 0), datetime.datetime(2024, 2, 9, 0, 0), datetime.datetime(2024, 2, 10, 0, 0), datetime.datetime(2024, 2, 11, 0, 0), datetime.datetime(2024, 2, 12, 0, 0), datetime.datetime(2024, 2, 13, 0, 0), datetime.datetime(2024, 2, 14, 0, 0), datetime.datetime(2024, 2, 15, 0, 0), datetime.datetime(2024, 2, 16, 0, 0), datetime.datetime(2024, 2, 17, 0, 0), datetime.datetime(2024, 2, 18, 0, 0), datetime.datetime(2024, 2, 19, 0, 0), datetime.datetime(2024, 2, 20, 0, 0), datetime.datetime(2024, 2, 21, 0, 0), datetime.datetime(2024, 2, 22, 0, 0), datetime.datetime(2024, 2, 23, 0, 0), datetime.datetime(2024, 2, 24, 0, 0), datetime.datetime(2024, 2, 25, 0, 0), datetime.datetime(2024, 2, 26, 0, 0), datetime.datetime(2024, 2, 27, 0, 0), datetime.datetime(2024, 2, 28, 0, 0), datetime.datetime(2024, 2, 29, 0, 0), datetime.datetime(2024, 3, 1, 0, 0), datetime.datetime(2024, 3, 2, 0, 0), datetime.datetime(2024, 3, 3, 0, 0), datetime.datetime(2024, 3, 4, 0, 0), datetime.datetime(2024, 3, 5, 0, 0), datetime.datetime(2024, 3, 6, 0, 0), datetime.datetime(2024, 3, 7, 0, 0), datetime.datetime(2024, 3, 8, 0, 0), datetime.datetime(2024, 3, 9, 0, 0), datetime.datetime(2024, 3, 10, 0, 0), datetime.datetime(2024, 3, 11, 0, 0), datetime.datetime(2024, 3, 12, 0, 0), datetime.datetime(2024, 3, 13, 0, 0), datetime.datetime(2024, 3, 14, 0, 0), datetime.datetime(2024, 3, 15, 0, 0), datetime.datetime(2024, 3, 16, 0, 0), datetime.datetime(2024, 3, 17, 0, 0), datetime.datetime(2024, 3, 18, 0, 0), datetime.datetime(2024, 3, 19, 0, 0), datetime.datetime(2024, 3, 20, 0, 0), datetime.datetime(2024, 3, 21, 0, 0), datetime.datetime(2024, 3, 22, 0, 0), datetime.datetime(2024, 3, 23, 0, 0), datetime.datetime(2024, 3, 24, 0, 0), datetime.datetime(2024, 3, 25, 0, 0), datetime.datetime(2024, 3, 26, 0, 0), datetime.datetime(2024, 3, 27, 0, 0), datetime.datetime(2024, 3, 28, 0, 0), datetime.datetime(2024, 3, 29, 0, 0), datetime.datetime(2024, 3, 30, 0, 0), datetime.datetime(2024, 3, 31, 0, 0), datetime.datetime(2024, 4, 1, 0, 0), datetime.datetime(2024, 4, 2, 0, 0), datetime.datetime(2024, 4, 3, 0, 0), datetime.datetime(2024, 4, 4, 0, 0), datetime.datetime(2024, 4, 5, 0, 0), datetime.datetime(2024, 4, 6, 0, 0), datetime.datetime(2024, 4, 7, 0, 0), datetime.datetime(2024, 4, 8, 0, 0), datetime.datetime(2024, 4, 9, 0, 0), datetime.datetime(2024, 4, 10, 0, 0), datetime.datetime(2024, 4, 11, 0, 0), datetime.datetime(2024, 4, 12, 0, 0), datetime.datetime(2024, 4, 13, 0, 0), datetime.datetime(2024, 4, 14, 0, 0), datetime.datetime(2024, 4, 15, 0, 0), datetime.datetime(2024, 4, 16, 0, 0), datetime.datetime(2024, 4, 17, 0, 0), datetime.datetime(2024, 4, 18, 0, 0), datetime.datetime(2024, 4, 19, 0, 0), datetime.datetime(2024, 4, 20, 0, 0), datetime.datetime(2024, 4, 21, 0, 0), datetime.datetime(2024, 4, 22, 0, 0), datetime.datetime(2024, 4, 23, 0, 0), datetime.datetime(2024, 4, 24, 0, 0), datetime.datetime(2024, 4, 25, 0, 0), datetime.datetime(2024, 4, 26, 0, 0), datetime.datetime(2024, 4, 27, 0, 0), datetime.datetime(2024, 4, 28, 0, 0), datetime.datetime(2024, 4, 29, 0, 0), datetime.datetime(2024, 4, 30, 0, 0), datetime.datetime(2024, 5, 1, 0, 0), datetime.datetime(2024, 5, 2, 0, 0), datetime.datetime(2024, 5, 3, 0, 0), datetime.datetime(2024, 5, 4, 0, 0), datetime.datetime(2024, 5, 5, 0, 0), datetime.datetime(2024, 5, 6, 0, 0), datetime.datetime(2024, 5, 7, 0, 0), datetime.datetime(2024, 5, 8, 0, 0), datetime.datetime(2024, 5, 9, 0, 0), datetime.datetime(2024, 5, 10, 0, 0), datetime.datetime(2024, 5, 11, 0, 0), datetime.datetime(2024, 5, 12, 0, 0), datetime.datetime(2024, 5, 13, 0, 0), datetime.datetime(2024, 5, 14, 0, 0), datetime.datetime(2024, 5, 15, 0, 0), datetime.datetime(2024, 5, 16, 0, 0), datetime.datetime(2024, 5, 17, 0, 0), datetime.datetime(2024, 5, 18, 0, 0), datetime.datetime(2024, 5, 19, 0, 0), datetime.datetime(2024, 5, 20, 0, 0), datetime.datetime(2024, 5, 21, 0, 0), datetime.datetime(2024, 5, 22, 0, 0), datetime.datetime(2024, 5, 23, 0, 0), datetime.datetime(2024, 5, 24, 0, 0), datetime.datetime(2024, 5, 25, 0, 0), datetime.datetime(2024, 5, 26, 0, 0), datetime.datetime(2024, 5, 27, 0, 0), datetime.datetime(2024, 5, 28, 0, 0), datetime.datetime(2024, 5, 29, 0, 0), datetime.datetime(2024, 5, 30, 0, 0), datetime.datetime(2024, 5, 31, 0, 0), datetime.datetime(2024, 6, 1, 0, 0), datetime.datetime(2024, 6, 2, 0, 0), datetime.datetime(2024, 6, 3, 0, 0), datetime.datetime(2024, 6, 4, 0, 0), datetime.datetime(2024, 6, 5, 0, 0), datetime.datetime(2024, 6, 6, 0, 0), datetime.datetime(2024, 6, 7, 0, 0), datetime.datetime(2024, 6, 8, 0, 0), datetime.datetime(2024, 6, 9, 0, 0), datetime.datetime(2024, 6, 10, 0, 0), datetime.datetime(2024, 6, 11, 0, 0), datetime.datetime(2024, 6, 12, 0, 0), datetime.datetime(2024, 6, 13, 0, 0), datetime.datetime(2024, 6, 14, 0, 0), datetime.datetime(2024, 6, 15, 0, 0), datetime.datetime(2024, 6, 16, 0, 0), datetime.datetime(2024, 6, 17, 0, 0), datetime.datetime(2024, 6, 18, 0, 0), datetime.datetime(2024, 6, 19, 0, 0), datetime.datetime(2024, 6, 20, 0, 0), datetime.datetime(2024, 6, 21, 0, 0), datetime.datetime(2024, 6, 22, 0, 0), datetime.datetime(2024, 6, 23, 0, 0), datetime.datetime(2024, 6, 24, 0, 0), datetime.datetime(2024, 6, 25, 0, 0), datetime.datetime(2024, 6, 26, 0, 0), datetime.datetime(2024, 6, 27, 0, 0), datetime.datetime(2024, 6, 28, 0, 0), datetime.datetime(2024, 6, 29, 0, 0), datetime.datetime(2024, 6, 30, 0, 0), datetime.datetime(2024, 7, 1, 0, 0), datetime.datetime(2024, 7, 2, 0, 0), datetime.datetime(2024, 7, 3, 0, 0), datetime.datetime(2024, 7, 4, 0, 0), datetime.datetime(2024, 7, 5, 0, 0), datetime.datetime(2024, 7, 6, 0, 0), datetime.datetime(2024, 7, 7, 0, 0), datetime.datetime(2024, 7, 8, 0, 0), datetime.datetime(2024, 7, 9, 0, 0), datetime.datetime(2024, 7, 10, 0, 0), datetime.datetime(2024, 7, 11, 0, 0), datetime.datetime(2024, 7, 12, 0, 0), datetime.datetime(2024, 7, 13, 0, 0), datetime.datetime(2024, 7, 14, 0, 0), datetime.datetime(2024, 7, 15, 0, 0), datetime.datetime(2024, 7, 16, 0, 0), datetime.datetime(2024, 7, 17, 0, 0), datetime.datetime(2024, 7, 18, 0, 0), datetime.datetime(2024, 7, 19, 0, 0), datetime.datetime(2024, 7, 20, 0, 0), datetime.datetime(2024, 7, 21, 0, 0), datetime.datetime(2024, 7, 22, 0, 0), datetime.datetime(2024, 7, 23, 0, 0), datetime.datetime(2024, 7, 24, 0, 0), datetime.datetime(2024, 7, 25, 0, 0), datetime.datetime(2024, 7, 26, 0, 0), datetime.datetime(2024, 7, 27, 0, 0), datetime.datetime(2024, 7, 28, 0, 0), datetime.datetime(2024, 7, 29, 0, 0), datetime.datetime(2024, 7, 30, 0, 0), datetime.datetime(2024, 7, 31, 0, 0), datetime.datetime(2024, 8, 1, 0, 0), datetime.datetime(2024, 8, 2, 0, 0), datetime.datetime(2024, 8, 3, 0, 0), datetime.datetime(2024, 8, 4, 0, 0), datetime.datetime(2024, 8, 5, 0, 0), datetime.datetime(2024, 8, 6, 0, 0), datetime.datetime(2024, 8, 7, 0, 0), datetime.datetime(2024, 8, 8, 0, 0), datetime.datetime(2024, 8, 9, 0, 0), datetime.datetime(2024, 8, 10, 0, 0), datetime.datetime(2024, 8, 11, 0, 0), datetime.datetime(2024, 8, 12, 0, 0), datetime.datetime(2024, 8, 13, 0, 0), datetime.datetime(2024, 8, 14, 0, 0), datetime.datetime(2024, 8, 15, 0, 0), datetime.datetime(2024, 8, 16, 0, 0), datetime.datetime(2024, 8, 17, 0, 0), datetime.datetime(2024, 8, 18, 0, 0), datetime.datetime(2024, 8, 19, 0, 0), datetime.datetime(2024, 8, 20, 0, 0), datetime.datetime(2024, 8, 21, 0, 0), datetime.datetime(2024, 8, 22, 0, 0), datetime.datetime(2024, 8, 23, 0, 0), datetime.datetime(2024, 8, 24, 0, 0), datetime.datetime(2024, 8, 25, 0, 0), datetime.datetime(2024, 8, 26, 0, 0), datetime.datetime(2024, 8, 27, 0, 0), datetime.datetime(2024, 8, 28, 0, 0), datetime.datetime(2024, 8, 29, 0, 0), datetime.datetime(2024, 8, 30, 0, 0), datetime.datetime(2024, 8, 31, 0, 0), datetime.datetime(2024, 9, 1, 0, 0), datetime.datetime(2024, 9, 2, 0, 0), datetime.datetime(2024, 9, 3, 0, 0), datetime.datetime(2024, 9, 4, 0, 0), datetime.datetime(2024, 9, 5, 0, 0), datetime.datetime(2024, 9, 6, 0, 0), datetime.datetime(2024, 9, 7, 0, 0), datetime.datetime(2024, 9, 8, 0, 0), datetime.datetime(2024, 9, 9, 0, 0), datetime.datetime(2024, 9, 10, 0, 0), datetime.datetime(2024, 9, 11, 0, 0), datetime.datetime(2024, 9, 12, 0, 0), datetime.datetime(2024, 9, 13, 0, 0), datetime.datetime(2024, 9, 14, 0, 0), datetime.datetime(2024, 9, 15, 0, 0), datetime.datetime(2024, 9, 16, 0, 0), datetime.datetime(2024, 9, 17, 0, 0), datetime.datetime(2024, 9, 18, 0, 0), datetime.datetime(2024, 9, 19, 0, 0), datetime.datetime(2024, 9, 20, 0, 0), datetime.datetime(2024, 9, 21, 0, 0), datetime.datetime(2024, 9, 22, 0, 0), datetime.datetime(2024, 9, 23, 0, 0), datetime.datetime(2024, 9, 24, 0, 0), datetime.datetime(2024, 9, 25, 0, 0), datetime.datetime(2024, 9, 26, 0, 0), datetime.datetime(2024, 9, 27, 0, 0), datetime.datetime(2024, 9, 28, 0, 0), datetime.datetime(2024, 9, 29, 0, 0), datetime.datetime(2024, 9, 30, 0, 0), datetime.datetime(2024, 10, 1, 0, 0), datetime.datetime(2024, 10, 2, 0, 0), datetime.datetime(2024, 10, 3, 0, 0), datetime.datetime(2024, 10, 4, 0, 0), datetime.datetime(2024, 10, 5, 0, 0), datetime.datetime(2024, 10, 6, 0, 0), datetime.datetime(2024, 10, 7, 0, 0), datetime.datetime(2024, 10, 8, 0, 0), datetime.datetime(2024, 10, 9, 0, 0), datetime.datetime(2024, 10, 10, 0, 0), datetime.datetime(2024, 10, 11, 0, 0), datetime.datetime(2024, 10, 12, 0, 0), datetime.datetime(2024, 10, 13, 0, 0), datetime.datetime(2024, 10, 14, 0, 0), datetime.datetime(2024, 10, 15, 0, 0), datetime.datetime(2024, 10, 16, 0, 0), datetime.datetime(2024, 10, 17, 0, 0), datetime.datetime(2024, 10, 18, 0, 0), datetime.datetime(2024, 10, 19, 0, 0), datetime.datetime(2024, 10, 20, 0, 0), datetime.datetime(2024, 10, 21, 0, 0), datetime.datetime(2024, 10, 22, 0, 0), datetime.datetime(2024, 10, 23, 0, 0), datetime.datetime(2024, 10, 24, 0, 0), datetime.datetime(2024, 10, 25, 0, 0), datetime.datetime(2024, 10, 26, 0, 0), datetime.datetime(2024, 10, 27, 0, 0), datetime.datetime(2024, 10, 28, 0, 0), datetime.datetime(2024, 10, 29, 0, 0), datetime.datetime(2024, 10, 30, 0, 0), datetime.datetime(2024, 10, 31, 0, 0), datetime.datetime(2024, 11, 1, 0, 0), datetime.datetime(2024, 11, 2, 0, 0), datetime.datetime(2024, 11, 3, 0, 0), datetime.datetime(2024, 11, 4, 0, 0), datetime.datetime(2024, 11, 5, 0, 0), datetime.datetime(2024, 11, 6, 0, 0), datetime.datetime(2024, 11, 7, 0, 0), datetime.datetime(2024, 11, 8, 0, 0), datetime.datetime(2024, 11, 9, 0, 0), datetime.datetime(2024, 11, 10, 0, 0), datetime.datetime(2024, 11, 11, 0, 0), datetime.datetime(2024, 11, 12, 0, 0), datetime.datetime(2024, 11, 13, 0, 0), datetime.datetime(2024, 11, 14, 0, 0), datetime.datetime(2024, 11, 15, 0, 0), datetime.datetime(2024, 11, 16, 0, 0), datetime.datetime(2024, 11, 17, 0, 0), datetime.datetime(2024, 11, 18, 0, 0), datetime.datetime(2024, 11, 19, 0, 0), datetime.datetime(2024, 11, 20, 0, 0), datetime.datetime(2024, 11, 21, 0, 0), datetime.datetime(2024, 11, 22, 0, 0), datetime.datetime(2024, 11, 23, 0, 0), datetime.datetime(2024, 11, 24, 0, 0), datetime.datetime(2024, 11, 25, 0, 0), datetime.datetime(2024, 11, 26, 0, 0), datetime.datetime(2024, 11, 27, 0, 0), datetime.datetime(2024, 11, 28, 0, 0), datetime.datetime(2024, 11, 29, 0, 0), datetime.datetime(2024, 11, 30, 0, 0), datetime.datetime(2024, 12, 1, 0, 0), datetime.datetime(2024, 12, 2, 0, 0), datetime.datetime(2024, 12, 3, 0, 0), datetime.datetime(2024, 12, 4, 0, 0), datetime.datetime(2024, 12, 5, 0, 0), datetime.datetime(2024, 12, 6, 0, 0), datetime.datetime(2024, 12, 7, 0, 0), datetime.datetime(2024, 12, 8, 0, 0), datetime.datetime(2024, 12, 9, 0, 0), datetime.datetime(2024, 12, 10, 0, 0), datetime.datetime(2024, 12, 11, 0, 0), datetime.datetime(2024, 12, 12, 0, 0), datetime.datetime(2024, 12, 13, 0, 0), datetime.datetime(2024, 12, 14, 0, 0), datetime.datetime(2024, 12, 15, 0, 0), datetime.datetime(2024, 12, 16, 0, 0), datetime.datetime(2024, 12, 17, 0, 0), datetime.datetime(2024, 12, 18, 0, 0), datetime.datetime(2024, 12, 19, 0, 0), datetime.datetime(2024, 12, 20, 0, 0), datetime.datetime(2024, 12, 21, 0, 0), datetime.datetime(2024, 12, 22, 0, 0), datetime.datetime(2024, 12, 23, 0, 0), datetime.datetime(2024, 12, 24, 0, 0), datetime.datetime(2024, 12, 25, 0, 0), datetime.datetime(2024, 12, 26, 0, 0), datetime.datetime(2024, 12, 27, 0, 0), datetime.datetime(2024, 12, 28, 0, 0), datetime.datetime(2024, 12, 29, 0, 0), datetime.datetime(2024, 12, 30, 0, 0), datetime.datetime(2024, 12, 31, 0, 0), datetime.datetime(2025, 1, 1, 0, 0), datetime.datetime(2025, 1, 2, 0, 0), datetime.datetime(2025, 1, 3, 0, 0), datetime.datetime(2025, 1, 4, 0, 0), datetime.datetime(2025, 1, 5, 0, 0), datetime.datetime(2025, 1, 6, 0, 0), datetime.datetime(2025, 1, 7, 0, 0), datetime.datetime(2025, 1, 8, 0, 0), datetime.datetime(2025, 1, 9, 0, 0), datetime.datetime(2025, 1, 10, 0, 0), datetime.datetime(2025, 1, 11, 0, 0), datetime.datetime(2025, 1, 12, 0, 0), datetime.datetime(2025, 1, 13, 0, 0), datetime.datetime(2025, 1, 14, 0, 0), datetime.datetime(2025, 1, 15, 0, 0), datetime.datetime(2025, 1, 16, 0, 0), datetime.datetime(2025, 1, 17, 0, 0), datetime.datetime(2025, 1, 18, 0, 0), datetime.datetime(2025, 1, 19, 0, 0), datetime.datetime(2025, 1, 20, 0, 0), datetime.datetime(2025, 1, 21, 0, 0), datetime.datetime(2025, 1, 22, 0, 0), datetime.datetime(2025, 1, 23, 0, 0), datetime.datetime(2025, 1, 24, 0, 0), datetime.datetime(2025, 1, 25, 0, 0), datetime.datetime(2025, 1, 26, 0, 0), datetime.datetime(2025, 1, 27, 0, 0), datetime.datetime(2025, 1, 28, 0, 0), datetime.datetime(2025, 1, 29, 0, 0), datetime.datetime(2025, 1, 30, 0, 0), datetime.datetime(2025, 1, 31, 0, 0), datetime.datetime(2025, 2, 1, 0, 0), datetime.datetime(2025, 2, 2, 0, 0), datetime.datetime(2025, 2, 3, 0, 0), datetime.datetime(2025, 2, 4, 0, 0), datetime.datetime(2025, 2, 5, 0, 0), datetime.datetime(2025, 2, 6, 0, 0), datetime.datetime(2025, 2, 7, 0, 0), datetime.datetime(2025, 2, 8, 0, 0), datetime.datetime(2025, 2, 9, 0, 0), datetime.datetime(2025, 2, 10, 0, 0), datetime.datetime(2025, 2, 11, 0, 0), datetime.datetime(2025, 2, 12, 0, 0), datetime.datetime(2025, 2, 13, 0, 0), datetime.datetime(2025, 2, 14, 0, 0), datetime.datetime(2025, 2, 15, 0, 0), datetime.datetime(2025, 2, 16, 0, 0), datetime.datetime(2025, 2, 17, 0, 0), datetime.datetime(2025, 2, 18, 0, 0), datetime.datetime(2025, 2, 19, 0, 0), datetime.datetime(2025, 2, 20, 0, 0), datetime.datetime(2025, 2, 21, 0, 0), datetime.datetime(2025, 2, 22, 0, 0), datetime.datetime(2025, 2, 23, 0, 0), datetime.datetime(2025, 2, 24, 0, 0), datetime.datetime(2025, 2, 25, 0, 0), datetime.datetime(2025, 2, 26, 0, 0), datetime.datetime(2025, 2, 27, 0, 0), datetime.datetime(2025, 2, 28, 0, 0), datetime.datetime(2025, 3, 1, 0, 0), datetime.datetime(2025, 3, 2, 0, 0), datetime.datetime(2025, 3, 3, 0, 0), datetime.datetime(2025, 3, 4, 0, 0), datetime.datetime(2025, 3, 5, 0, 0), datetime.datetime(2025, 3, 6, 0, 0), datetime.datetime(2025, 3, 7, 0, 0), datetime.datetime(2025, 3, 8, 0, 0), datetime.datetime(2025, 3, 9, 0, 0), datetime.datetime(2025, 3, 10, 0, 0), datetime.datetime(2025, 3, 11, 0, 0), datetime.datetime(2025, 3, 12, 0, 0), datetime.datetime(2025, 3, 13, 0, 0), datetime.datetime(2025, 3, 14, 0, 0), datetime.datetime(2025, 3, 15, 0, 0), datetime.datetime(2025, 3, 16, 0, 0), datetime.datetime(2025, 3, 17, 0, 0), datetime.datetime(2025, 3, 18, 0, 0), datetime.datetime(2025, 3, 19, 0, 0), datetime.datetime(2025, 3, 20, 0, 0), datetime.datetime(2025, 3, 21, 0, 0), datetime.datetime(2025, 3, 22, 0, 0), datetime.datetime(2025, 3, 23, 0, 0), datetime.datetime(2025, 3, 24, 0, 0), datetime.datetime(2025, 3, 25, 0, 0), datetime.datetime(2025, 3, 26, 0, 0), datetime.datetime(2025, 3, 27, 0, 0), datetime.datetime(2025, 3, 28, 0, 0), datetime.datetime(2025, 3, 29, 0, 0), datetime.datetime(2025, 3, 30, 0, 0), datetime.datetime(2025, 3, 31, 0, 0), datetime.datetime(2025, 4, 1, 0, 0), datetime.datetime(2025, 4, 2, 0, 0), datetime.datetime(2025, 4, 3, 0, 0), datetime.datetime(2025, 4, 4, 0, 0), datetime.datetime(2025, 4, 5, 0, 0), datetime.datetime(2025, 4, 6, 0, 0), datetime.datetime(2025, 4, 7, 0, 0), datetime.datetime(2025, 4, 8, 0, 0), datetime.datetime(2025, 4, 9, 0, 0), datetime.datetime(2025, 4, 10, 0, 0), datetime.datetime(2025, 4, 11, 0, 0), datetime.datetime(2025, 4, 12, 0, 0), datetime.datetime(2025, 4, 13, 0, 0), datetime.datetime(2025, 4, 14, 0, 0), datetime.datetime(2025, 4, 15, 0, 0), datetime.datetime(2025, 4, 16, 0, 0), datetime.datetime(2025, 4, 17, 0, 0), datetime.datetime(2025, 4, 18, 0, 0), datetime.datetime(2025, 4, 19, 0, 0), datetime.datetime(2025, 4, 20, 0, 0), datetime.datetime(2025, 4, 21, 0, 0), datetime.datetime(2025, 4, 22, 0, 0), datetime.datetime(2025, 4, 23, 0, 0), datetime.datetime(2025, 4, 24, 0, 0), datetime.datetime(2025, 4, 25, 0, 0), datetime.datetime(2025, 4, 26, 0, 0), datetime.datetime(2025, 4, 27, 0, 0), datetime.datetime(2025, 4, 28, 0, 0), datetime.datetime(2025, 4, 29, 0, 0), datetime.datetime(2025, 4, 30, 0, 0), datetime.datetime(2025, 5, 1, 0, 0), datetime.datetime(2025, 5, 2, 0, 0), datetime.datetime(2025, 5, 3, 0, 0), datetime.datetime(2025, 5, 4, 0, 0), datetime.datetime(2025, 5, 5, 0, 0), datetime.datetime(2025, 5, 6, 0, 0), datetime.datetime(2025, 5, 7, 0, 0), datetime.datetime(2025, 5, 8, 0, 0), datetime.datetime(2025, 5, 9, 0, 0), datetime.datetime(2025, 5, 10, 0, 0), datetime.datetime(2025, 5, 11, 0, 0), datetime.datetime(2025, 5, 12, 0, 0), datetime.datetime(2025, 5, 13, 0, 0), datetime.datetime(2025, 5, 14, 0, 0), datetime.datetime(2025, 5, 15, 0, 0), datetime.datetime(2025, 5, 16, 0, 0), datetime.datetime(2025, 5, 17, 0, 0), datetime.datetime(2025, 5, 18, 0, 0), datetime.datetime(2025, 5, 19, 0, 0), datetime.datetime(2025, 5, 20, 0, 0), datetime.datetime(2025, 5, 21, 0, 0), datetime.datetime(2025, 5, 22, 0, 0), datetime.datetime(2025, 5, 23, 0, 0), datetime.datetime(2025, 5, 24, 0, 0), datetime.datetime(2025, 5, 25, 0, 0), datetime.datetime(2025, 5, 26, 0, 0), datetime.datetime(2025, 5, 27, 0, 0), datetime.datetime(2025, 5, 28, 0, 0), datetime.datetime(2025, 5, 29, 0, 0), datetime.datetime(2025, 5, 30, 0, 0), datetime.datetime(2025, 5, 31, 0, 0), datetime.datetime(2025, 6, 1, 0, 0), datetime.datetime(2025, 6, 2, 0, 0), datetime.datetime(2025, 6, 3, 0, 0), datetime.datetime(2025, 6, 4, 0, 0), datetime.datetime(2025, 6, 5, 0, 0), datetime.datetime(2025, 6, 6, 0, 0), datetime.datetime(2025, 6, 7, 0, 0), datetime.datetime(2025, 6, 8, 0, 0), datetime.datetime(2025, 6, 9, 0, 0), datetime.datetime(2025, 6, 10, 0, 0), datetime.datetime(2025, 6, 11, 0, 0), datetime.datetime(2025, 6, 12, 0, 0), datetime.datetime(2025, 6, 13, 0, 0), datetime.datetime(2025, 6, 14, 0, 0), datetime.datetime(2025, 6, 15, 0, 0), datetime.datetime(2025, 6, 16, 0, 0), datetime.datetime(2025, 6, 17, 0, 0), datetime.datetime(2025, 6, 18, 0, 0), datetime.datetime(2025, 6, 19, 0, 0), datetime.datetime(2025, 6, 20, 0, 0), datetime.datetime(2025, 6, 21, 0, 0), datetime.datetime(2025, 6, 22, 0, 0), datetime.datetime(2025, 6, 23, 0, 0), datetime.datetime(2025, 6, 24, 0, 0), datetime.datetime(2025, 6, 25, 0, 0), datetime.datetime(2025, 6, 26, 0, 0), datetime.datetime(2025, 6, 27, 0, 0), datetime.datetime(2025, 6, 28, 0, 0), datetime.datetime(2025, 6, 29, 0, 0), datetime.datetime(2025, 6, 30, 0, 0), datetime.datetime(2025, 7, 1, 0, 0), datetime.datetime(2025, 7, 2, 0, 0), datetime.datetime(2025, 7, 3, 0, 0), datetime.datetime(2025, 7, 4, 0, 0), datetime.datetime(2025, 7, 5, 0, 0), datetime.datetime(2025, 7, 6, 0, 0), datetime.datetime(2025, 7, 7, 0, 0), datetime.datetime(2025, 7, 8, 0, 0), datetime.datetime(2025, 7, 9, 0, 0), datetime.datetime(2025, 7, 10, 0, 0), datetime.datetime(2025, 7, 11, 0, 0), datetime.datetime(2025, 7, 12, 0, 0), datetime.datetime(2025, 7, 13, 0, 0), datetime.datetime(2025, 7, 14, 0, 0), datetime.datetime(2025, 7, 15, 0, 0), datetime.datetime(2025, 7, 16, 0, 0), datetime.datetime(2025, 7, 17, 0, 0), datetime.datetime(2025, 7, 18, 0, 0), datetime.datetime(2025, 7, 19, 0, 0), datetime.datetime(2025, 7, 20, 0, 0), datetime.datetime(2025, 7, 21, 0, 0), datetime.datetime(2025, 7, 22, 0, 0), datetime.datetime(2025, 7, 23, 0, 0), datetime.datetime(2025, 7, 24, 0, 0), datetime.datetime(2025, 7, 25, 0, 0), datetime.datetime(2025, 7, 26, 0, 0), datetime.datetime(2025, 7, 27, 0, 0), datetime.datetime(2025, 7, 28, 0, 0), datetime.datetime(2025, 7, 29, 0, 0), datetime.datetime(2025, 7, 30, 0, 0), datetime.datetime(2025, 7, 31, 0, 0), datetime.datetime(2025, 8, 1, 0, 0), datetime.datetime(2025, 8, 2, 0, 0), datetime.datetime(2025, 8, 3, 0, 0), datetime.datetime(2025, 8, 4, 0, 0), datetime.datetime(2025, 8, 5, 0, 0), datetime.datetime(2025, 8, 6, 0, 0), datetime.datetime(2025, 8, 7, 0, 0), datetime.datetime(2025, 8, 8, 0, 0), datetime.datetime(2025, 8, 9, 0, 0), datetime.datetime(2025, 8, 10, 0, 0), datetime.datetime(2025, 8, 11, 0, 0), datetime.datetime(2025, 8, 12, 0, 0), datetime.datetime(2025, 8, 13, 0, 0), datetime.datetime(2025, 8, 14, 0, 0), datetime.datetime(2025, 8, 15, 0, 0), datetime.datetime(2025, 8, 16, 0, 0), datetime.datetime(2025, 8, 17, 0, 0), datetime.datetime(2025, 8, 18, 0, 0), datetime.datetime(2025, 8, 19, 0, 0), datetime.datetime(2025, 8, 20, 0, 0), datetime.datetime(2025, 8, 21, 0, 0), datetime.datetime(2025, 8, 22, 0, 0), datetime.datetime(2025, 8, 23, 0, 0), datetime.datetime(2025, 8, 24, 0, 0), datetime.datetime(2025, 8, 25, 0, 0), datetime.datetime(2025, 8, 26, 0, 0), datetime.datetime(2025, 8, 27, 0, 0), datetime.datetime(2025, 8, 28, 0, 0), datetime.datetime(2025, 8, 29, 0, 0), datetime.datetime(2025, 8, 30, 0, 0), datetime.datetime(2025, 8, 31, 0, 0), datetime.datetime(2025, 9, 1, 0, 0), datetime.datetime(2025, 9, 2, 0, 0), datetime.datetime(2025, 9, 3, 0, 0), datetime.datetime(2025, 9, 4, 0, 0), datetime.datetime(2025, 9, 5, 0, 0), datetime.datetime(2025, 9, 6, 0, 0), datetime.datetime(2025, 9, 7, 0, 0), datetime.datetime(2025, 9, 8, 0, 0), datetime.datetime(2025, 9, 9, 0, 0), datetime.datetime(2025, 9, 10, 0, 0), datetime.datetime(2025, 9, 11, 0, 0), datetime.datetime(2025, 9, 12, 0, 0), datetime.datetime(2025, 9, 13, 0, 0), datetime.datetime(2025, 9, 14, 0, 0), datetime.datetime(2025, 9, 15, 0, 0), datetime.datetime(2025, 9, 16, 0, 0), datetime.datetime(2025, 9, 17, 0, 0), datetime.datetime(2025, 9, 18, 0, 0), datetime.datetime(2025, 9, 19, 0, 0), datetime.datetime(2025, 9, 20, 0, 0), datetime.datetime(2025, 9, 21, 0, 0), datetime.datetime(2025, 9, 22, 0, 0), datetime.datetime(2025, 9, 23, 0, 0), datetime.datetime(2025, 9, 24, 0, 0), datetime.datetime(2025, 9, 25, 0, 0), datetime.datetime(2025, 9, 26, 0, 0), datetime.datetime(2025, 9, 27, 0, 0), datetime.datetime(2025, 9, 28, 0, 0), datetime.datetime(2025, 9, 29, 0, 0), datetime.datetime(2025, 9, 30, 0, 0), datetime.datetime(2025, 10, 1, 0, 0), datetime.datetime(2025, 10, 2, 0, 0), datetime.datetime(2025, 10, 3, 0, 0), datetime.datetime(2025, 10, 4, 0, 0), datetime.datetime(2025, 10, 5, 0, 0), datetime.datetime(2025, 10, 6, 0, 0), datetime.datetime(2025, 10, 7, 0, 0), datetime.datetime(2025, 10, 8, 0, 0), datetime.datetime(2025, 10, 9, 0, 0), datetime.datetime(2025, 10, 10, 0, 0), datetime.datetime(2025, 10, 11, 0, 0), datetime.datetime(2025, 10, 12, 0, 0), datetime.datetime(2025, 10, 13, 0, 0), datetime.datetime(2025, 10, 14, 0, 0), datetime.datetime(2025, 10, 15, 0, 0), datetime.datetime(2025, 10, 16, 0, 0), datetime.datetime(2025, 10, 17, 0, 0), datetime.datetime(2025, 10, 18, 0, 0), datetime.datetime(2025, 10, 19, 0, 0), datetime.datetime(2025, 10, 20, 0, 0), datetime.datetime(2025, 10, 21, 0, 0), datetime.datetime(2025, 10, 22, 0, 0), datetime.datetime(2025, 10, 23, 0, 0), datetime.datetime(2025, 10, 24, 0, 0), datetime.datetime(2025, 10, 25, 0, 0), datetime.datetime(2025, 10, 26, 0, 0), datetime.datetime(2025, 10, 27, 0, 0), datetime.datetime(2025, 10, 28, 0, 0), datetime.datetime(2025, 10, 29, 0, 0), datetime.datetime(2025, 10, 30, 0, 0), datetime.datetime(2025, 10, 31, 0, 0), datetime.datetime(2025, 11, 1, 0, 0), datetime.datetime(2025, 11, 2, 0, 0), datetime.datetime(2025, 11, 3, 0, 0), datetime.datetime(2025, 11, 4, 0, 0), datetime.datetime(2025, 11, 5, 0, 0), datetime.datetime(2025, 11, 6, 0, 0), datetime.datetime(2025, 11, 7, 0, 0), datetime.datetime(2025, 11, 8, 0, 0), datetime.datetime(2025, 11, 9, 0, 0), datetime.datetime(2025, 11, 10, 0, 0), datetime.datetime(2025, 11, 11, 0, 0), datetime.datetime(2025, 11, 12, 0, 0), datetime.datetime(2025, 11, 13, 0, 0), datetime.datetime(2025, 11, 14, 0, 0), datetime.datetime(2025, 11, 15, 0, 0), datetime.datetime(2025, 11, 16, 0, 0), datetime.datetime(2025, 11, 17, 0, 0), datetime.datetime(2025, 11, 18, 0, 0), datetime.datetime(2025, 11, 19, 0, 0), datetime.datetime(2025, 11, 20, 0, 0), datetime.datetime(2025, 11, 21, 0, 0), datetime.datetime(2025, 11, 22, 0, 0), datetime.datetime(2025, 11, 23, 0, 0), datetime.datetime(2025, 11, 24, 0, 0), datetime.datetime(2025, 11, 25, 0, 0), datetime.datetime(2025, 11, 26, 0, 0), datetime.datetime(2025, 11, 27, 0, 0), datetime.datetime(2025, 11, 28, 0, 0), datetime.datetime(2025, 11, 29, 0, 0), datetime.datetime(2025, 11, 30, 0, 0), datetime.datetime(2025, 12, 1, 0, 0), datetime.datetime(2025, 12, 2, 0, 0), datetime.datetime(2025, 12, 3, 0, 0), datetime.datetime(2025, 12, 4, 0, 0), datetime.datetime(2025, 12, 5, 0, 0), datetime.datetime(2025, 12, 6, 0, 0), datetime.datetime(2025, 12, 7, 0, 0), datetime.datetime(2025, 12, 8, 0, 0), datetime.datetime(2025, 12, 9, 0, 0), datetime.datetime(2025, 12, 10, 0, 0), datetime.datetime(2025, 12, 11, 0, 0), datetime.datetime(2025, 12, 12, 0, 0), datetime.datetime(2025, 12, 13, 0, 0), datetime.datetime(2025, 12, 14, 0, 0), datetime.datetime(2025, 12, 15, 0, 0), datetime.datetime(2025, 12, 16, 0, 0), datetime.datetime(2025, 12, 17, 0, 0), datetime.datetime(2025, 12, 18, 0, 0), datetime.datetime(2025, 12, 19, 0, 0), datetime.datetime(2025, 12, 20, 0, 0), datetime.datetime(2025, 12, 21, 0, 0), datetime.datetime(2025, 12, 22, 0, 0), datetime.datetime(2025, 12, 23, 0, 0), datetime.datetime(2025, 12, 24, 0, 0), datetime.datetime(2025, 12, 25, 0, 0), datetime.datetime(2025, 12, 26, 0, 0), datetime.datetime(2025, 12, 27, 0, 0), datetime.datetime(2025, 12, 28, 0, 0), datetime.datetime(2025, 12, 29, 0, 0), datetime.datetime(2025, 12, 30, 0, 0), datetime.datetime(2025, 12, 31, 0, 0)]
    algo_q = parse_masked_array_text(algo_q)
    algo_t = parse_masked_array_text(algo_t)
    station_q = parse_masked_array_text(station_q)
    if isinstance(station_date, str):
        station_date = parse_datetime_list_text(station_date)
    indicators_list = ['spearman']
    indicators_list = None
    indicators = run_indicators(y_true_val=station_q, y_true_dim=station_date, y_pred_val=algo_q, y_pred_dim=algo_t, min_dates=10, indicators_list=indicators_list)
    print('INDICATORS:', indicators)
if __name__ == '__main__':
    main()