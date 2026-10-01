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

@authors: Callum TYLER, callum.tyler@inrae.fr
          Hind OUBANAS, hind.oubanas@inrae.fr
          Nikki TEBALDI, ntebaldi@umass.edu
          Dylan QUITTARD, dylan.quittard@inrae.fr
            All functions not mentioned above.
          Cécile Cazals, cecile.cazals@cs-soprasteria.com
            logger

Description:
    This file gather functions used from sic4dvar.py
"""
import copy
import json
import logging
import os
import sys
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import sic4dvar_params as params
from lib.lib_config import read_config
from lib.lib_dates import convert_to_YMD, count_days_in_months, date_to_seconds, seconds_to_date_old, daynum_to_date
from lib.lib_indicators import compute_all_indicators_from_predict_true
from lib.lib_log import call_error_message
from lib.lib_netcdf import get_nc_variable_data
from lib.lib_sos import get_station_q_and_qt
from lib.lib_verif import check_na, reorder_ids_with_indices
from sic4dvar_functions import sic4dvar_calculations as calc
from sic4dvar_functions.sic4dvar_gnuplot_save import gnuplot_save_q
from sic4dvar_functions.helpers.helpers_arrays import find_n_nearest, find_nearest, get_index_valid_data, get_mask_nan_across_arrays, masked_array_to_nan_array, nan_array_to_masked_array
from sic4dvar_functions.Y786 import K
from sic4dvar_functions.sic4dvar_calculations import verify_name_length
from lib.lib_dates import seconds_to_time_str
from sic4dvar_modules.sic4dvar_compute_slope_and_bathymetry import aggregate_node_bathy_to_reach, compute_wet_area

def read_discharge_from_npy(reach_id, path):
    try:
        q = np.load(path + str(reach_id) + '.npy')
    except:
        q = []
    return q

def algo5_fill_removed_data(computed_q, removed_indices, full_size):
    j = 0
    q_est = []
    for i in range(0, full_size):
        if i in removed_indices:
            q_est.append(np.nan)
        else:
            q_est.append(computed_q[j])
            j += 1
    mask_q_est = np.ones(len(q_est), dtype=bool)
    for i in range(len(q_est)):
        if not calc.check_na(q_est[i]):
            mask_q_est[i] = False
    q_est_masked = np.ma.array(q_est, mask=mask_q_est)
    return q_est_masked

def large_deviations_removal(node_x, z):
    new_z = deepcopy(z)
    for i in range(0, 1):
        for t in range(0, len(z[0])):
            index_valid = []
            for n in range(0, len(z)):
                if not calc.check_na(z[n, t]):
                    index_valid.append(n)
            num_valid_pts = len(index_valid)
            num_overpass = np.arange(0, len(z[0]))
            if num_valid_pts > 2:
                a11 = 0.0
                a12 = 0.0
                a21 = 0.0
                a22 = 0.0
                b1 = 0.0
                b2 = 0.0
                for n in range(0, num_valid_pts):
                    a11 = a11 + node_x[index_valid[n]] * node_x[index_valid[n]]
                    a12 = a12 + node_x[index_valid[n]]
                    a21 = a21 + node_x[index_valid[n]]
                    a22 = a22 + 1.0
                    b1 = b1 + node_x[index_valid[n]] * z[index_valid[n], t]
                    b2 = b2 + z[index_valid[n], t]
                c2 = (b1 * a21 - b2 * a11) / (a12 * a21 - a22 * a11)
                c1 = (b1 - a12 * c2) / a11
                c3 = 0.0
                for n in range(0, num_valid_pts):
                    dist = c1 * node_x[index_valid[n]] + c2 - z[index_valid[n], t]
                    c3 = c3 + dist ** 2
                c3 = np.sqrt(c3 / num_valid_pts)
                for n in range(0, num_valid_pts):
                    dist = c1 * node_x[index_valid[n]] + c2 - z[index_valid[n], t]
                    if abs(dist) > 2.0 * c3:
                        new_z[index_valid[n], t] = c1 * node_x[index_valid[n]] + c2
    return new_z

def global_large_deviations_removal_relative(node_x, z):
    z_writable = np.array(z, copy=True)
    z_star = np.nanpercentile(z_writable, 0.05)
    node_x = abs(node_x - node_x[0]) / 1000
    new_z = deepcopy(z)
    a11 = 0.0
    a12 = 0.0
    a21 = 0.0
    a22 = 0.0
    b1 = 0.0
    b2 = 0.0
    c1 = 0.0
    c2 = 0.0
    total_valid_points = 0.0
    epsilon_p = 2.0
    epsilon_n = 2.0
    valid_indexes = []
    reverse_order = False
    for t in range(0, len(z[0])):
        index_valid = []
        for n in range(0, len(z)):
            if not calc.check_na(z[n, t]):
                index_valid.append(n)
        num_valid_pts = len(index_valid)
        total_valid_points += num_valid_pts
        num_overpass = np.arange(0, len(z[0]))
        valid_indexes.append(index_valid)
        for n in range(0, num_valid_pts):
            a11 = a11 + node_x[index_valid[n]] * node_x[index_valid[n]]
            a12 = a12 + node_x[index_valid[n]]
            a21 = a21 + node_x[index_valid[n]]
            a22 = a22 + 1.0
            b1 = b1 + node_x[index_valid[n]] * z[index_valid[n], t]
            b2 = b2 + z[index_valid[n], t]
    if total_valid_points >= 1:
        c2 = (b1 * a21 - b2 * a11) / (a12 * a21 - a22 * a11)
        c1 = (b1 - a12 * c2) / a11
    if c1 == 0.0 and c2 == 0.0:
        logging.warning("Couldn't remove outliers: no valid points found.")
        return (z, reverse_order)
    epsilon = 0.1
    x_min = 1000000.0
    x_max = 0.0
    for t in range(0, len(z[0])):
        if np.array(valid_indexes[t]).size > 0:
            x_min0 = node_x[valid_indexes[t][0]]
            x_max0 = node_x[valid_indexes[t][-1]]
            if x_min0 < x_min:
                x_min = x_min0
            if x_max0 > x_max:
                x_max = x_max0
    x_mean = (x_min + x_max) / 2.0
    c2_star = z_star - c1 * x_mean
    var_h = 0.0
    for t in range(0, len(z[0])):
        for n in range(0, len(valid_indexes[t])):
            var_h = var_h + ((new_z[valid_indexes[t][n], t] - (c1 * node_x[valid_indexes[t][n]] + c2_star)) / (c2 - c2_star)) ** 2
    var_h = np.sqrt(var_h / total_valid_points)
    for t in range(0, len(z[0])):
        for n in range(0, len(valid_indexes[t])):
            if (new_z[valid_indexes[t][n], t] - (c1 * node_x[valid_indexes[t][n]] + c2_star)) / (c2 - c2_star) > epsilon_p * var_h:
                new_z[valid_indexes[t][n], t] = c1 * node_x[valid_indexes[t][n]] + c2_star + (c2 - c2_star) * epsilon_p * var_h
            if (new_z[valid_indexes[t][n], t] - (c1 * node_x[valid_indexes[t][n]] + c2_star)) / (c2 - c2_star) < epsilon_n * var_h:
                new_z[valid_indexes[t][n], t] = c1 * node_x[valid_indexes[t][n]] + c2_star + (c2 - c2_star) * epsilon_n * var_h
    if c1 > 0.01:
        reverse_order = True
    return (new_z, reverse_order)

def global_large_deviations_removal(node_x, z, times_debug=np.array([])):
    debug_orig = True
    node_x = abs(node_x - node_x[0]) / 1000
    new_z = deepcopy(z)
    a11 = 0.0
    a12 = 0.0
    a21 = 0.0
    a22 = 0.0
    b1 = 0.0
    b2 = 0.0
    c1 = 0.0
    c2 = 0.0
    total_valid_points = 0.0
    for t in range(0, len(z[0])):
        index_valid = []
        for n in range(0, len(z)):
            if not calc.check_na(z[n, t]):
                index_valid.append(n)
        num_valid_pts = len(index_valid)
        total_valid_points += num_valid_pts
        num_overpass = np.arange(0, len(z[0]))
        for n in range(0, num_valid_pts):
            a11 = a11 + node_x[index_valid[n]] * node_x[index_valid[n]]
            a12 = a12 + node_x[index_valid[n]]
            a21 = a21 + node_x[index_valid[n]]
            a22 = a22 + 1.0
            b1 = b1 + node_x[index_valid[n]] * z[index_valid[n], t]
            b2 = b2 + z[index_valid[n], t]
    if total_valid_points >= 1:
        c2 = (b1 * a21 - b2 * a11) / (a12 * a21 - a22 * a11)
        c1 = (b1 - a12 * c2) / a11
    i_pslope = 0
    total_valid_points = 0.0
    if c1 > 0:
        i_pslope = 1
        ss0 = 0.0
        ss1 = 0.0
        for t in range(0, len(z[0])):
            index_valid = []
            for n in range(0, len(z)):
                if not calc.check_na(z[n, t]):
                    index_valid.append(n)
            num_valid_pts = len(index_valid)
            total_valid_points += num_valid_pts
            num_overpass = np.arange(0, len(z[0]))
            for n in range(0, num_valid_pts):
                ss0 = ss0 + c1 * node_x[index_valid[n]] + c2
                ss1 = ss1 + 1.0
        ss0 = ss0 / ss1
        c1 = 0.0
        c2 = ss0
    c3 = 0.0
    ss1 = 0.0
    total_valid_points = 0.0
    for t in range(0, len(z[0])):
        index_valid = []
        for n in range(0, len(z)):
            if not calc.check_na(z[n, t]):
                index_valid.append(n)
        num_valid_pts = len(index_valid)
        total_valid_points += num_valid_pts
        num_overpass = np.arange(0, len(z[0]))
        for n in range(0, num_valid_pts):
            dist = c1 * node_x[index_valid[n]] + c2 - z[index_valid[n], t]
            c3 = c3 + dist ** 2
            ss1 = ss1 + 1.0
    if total_valid_points >= 1:
        c3 = np.sqrt(c3 / ss1)
    sigmp = 2.0
    sigmn = 3.0
    if debug_orig:
        sigmn = 2.0
    if i_pslope == 1:
        sigmp = 1.0
        sigmn = 1.0
    total_valid_points = 0.0
    for t in range(0, len(z[0])):
        index_valid = []
        for n in range(0, len(z)):
            if not calc.check_na(z[n, t]):
                index_valid.append(n)
        num_valid_pts = len(index_valid)
        total_valid_points += num_valid_pts
        num_overpass = np.arange(0, len(z[0]))
        for n in range(0, num_valid_pts):
            dist = c1 * node_x[index_valid[n]] + c2 - z[index_valid[n], t]
            if dist > 0:
                if abs(dist) > sigmp * c3:
                    if debug_orig:
                        new_z[index_valid[n], t] = c1 * node_x[index_valid[n]] + c2 - dist * (1 * c3) / abs(dist)
                    else:
                        new_z[index_valid[n], t] = c1 * node_x[index_valid[n]] + c2 - dist * (sigmp * c3) / abs(dist)
            elif abs(dist) > sigmn * c3:
                if debug_orig:
                    new_z[index_valid[n], t] = c1 * node_x[index_valid[n]] + c2 - dist * (1 * c3) / abs(dist)
                else:
                    new_z[index_valid[n], t] = c1 * node_x[index_valid[n]] + c2 - dist * (sigmn * c3) / abs(dist)
    return (new_z, c1, c2)

def global_large_deviations_removal_experimental(node_x, z, reach_t, times_debug=np.array([])):
    reverse_order = False
    node_x = abs(node_x - node_x[0]) / 1000
    new_z = deepcopy(z)
    a11 = 0.0
    a12 = 0.0
    a21 = 0.0
    a22 = 0.0
    b1 = 0.0
    b2 = 0.0
    c1 = 0.0
    c2 = 0.0
    total_valid_points = 0.0
    time_acc = 0.0
    old_b1 = 0.0
    old_b2 = 0.0
    number_of_sections = []
    prev_index_valid = []
    for t in range(0, len(z[0])):
        index_valid = []
        for n in range(0, len(z)):
            if not calc.check_na(z[n, t]):
                index_valid.append(n)
        num_valid_pts = len(index_valid)
        total_valid_points += num_valid_pts
        num_overpass = np.arange(0, len(z[0]))
        time_acc_node = 0.0
        for n in range(0, num_valid_pts):
            a11 = a11 + node_x[index_valid[n]] * node_x[index_valid[n]]
            a12 = a12 + node_x[index_valid[n]]
            a21 = a21 + node_x[index_valid[n]]
            a22 = a22 + 1.0
            b1 = b1 + node_x[index_valid[n]] * z[index_valid[n], t]
            b2 = b2 + z[index_valid[n], t]
    if total_valid_points >= 1:
        if a12 * a21 - a22 * a11 == 0:
            logging.warning('a12*a21-a22*a11 == 0, cannot compute c1 and c2. Setting c1 and c2 to 0.')
            c1 = 0.0
            c2 = 0.0
        else:
            c2 = (b1 * a21 - b2 * a11) / (a12 * a21 - a22 * a11)
            c1 = (b1 - a12 * c2) / a11
    sigmp = 2.0
    sigmn = 2.0
    total_valid_points = 0.0
    error_array = []
    for t in range(0, len(z[0])):
        index_valid = []
        for n in range(0, len(z)):
            if not calc.check_na(z[n, t]):
                index_valid.append(n)
        num_valid_pts = len(index_valid)
        total_valid_points += num_valid_pts
        num_overpass = np.arange(0, len(z[0]))
        error_array_tmp = np.ones(num_valid_pts) * np.nan
        if num_valid_pts > 0:
            for n in range(0, num_valid_pts):
                dist = c1 * node_x[index_valid[n]] + c2 - z[index_valid[n], t]
                error_array_tmp[n] = dist
            error_array.append(error_array_tmp)
    error_array_concat = np.concatenate(error_array)
    from scipy.stats import iqr
    iqr_value = iqr(error_array_concat)
    median_value = np.nanmedian(error_array_concat)
    for t in range(0, len(z[0])):
        index_valid = []
        for n in range(0, len(z)):
            if not calc.check_na(z[n, t]):
                index_valid.append(n)
        num_valid_pts = len(index_valid)
        total_valid_points += num_valid_pts
        num_overpass = np.arange(0, len(z[0]))
        debug_dict = []
        debug_dict.append(0.0)
        debug_dict.append(0.0)
        debug_dict.append(0.0)
        for n in range(0, num_valid_pts):
            dist = c1 * node_x[index_valid[n]] + c2 - z[index_valid[n], t]
            if np.abs(debug_dict[2]) < np.abs(dist - median_value):
                debug_dict[0] = n
                debug_dict[1] = t
                debug_dict[2] = np.abs(dist - median_value)
            if np.abs(dist - median_value) > 1 * iqr_value:
                pass
            if np.abs(dist - median_value) > 3 * iqr_value:
                if dist - median_value > 0:
                    new_z[index_valid[n], t] = c1 * node_x[index_valid[n]] + c2 + (sigmp * iqr_value + median_value)
                else:
                    new_z[index_valid[n], t] = c1 * node_x[index_valid[n]] + c2 + (sigmn * iqr_value + median_value)
    if c1 > 0.01:
        reverse_order = True
    return (new_z, reverse_order, c1, c2)

def compute_mean_var_from_2D_array(test_swot_z_obs, reach_t):
    z_mean = []
    check_na_vector = np.vectorize(calc.check_na)
    for n in range(0, test_swot_z_obs.shape[0]):
        integrated_z_mean = 0.0
        time_scaling = 0.0
        for t in range(1, test_swot_z_obs.shape[1]):
            if calc.check_na(test_swot_z_obs[n, t]) or calc.check_na(test_swot_z_obs[n, t - 1]):
                integrated_z_mean += 0.0
                time_scaling += 0.0
            else:
                integrated_z_mean += (test_swot_z_obs[n, t] + test_swot_z_obs[n, t - 1]) / 2 * (reach_t[t] - reach_t[t - 1])
                time_scaling += reach_t[t] - reach_t[t - 1]
        if time_scaling != 0.0:
            integrated_z_mean = integrated_z_mean / time_scaling
        else:
            integrated_z_mean = -9999.0
        z_mean.append(integrated_z_mean)
    z_var = []
    for n in range(0, test_swot_z_obs.shape[0]):
        integrated_z_var = 0.0
        time_scaling = 0.0
        for t in range(1, test_swot_z_obs.shape[1]):
            if calc.check_na(test_swot_z_obs[n, t]) or calc.check_na(test_swot_z_obs[n, t - 1]):
                integrated_z_var += 0.0
                time_scaling += 0.0
            else:
                s0 = (test_swot_z_obs[n, t] - z_mean[n]) ** 2
                s1 = (test_swot_z_obs[n, t - 1] - z_mean[n]) ** 2
                integrated_z_var += (s0 + s1) / 2 * (reach_t[t] - reach_t[t - 1])
                time_scaling += reach_t[t] - reach_t[t - 1]
        if time_scaling != 0.0:
            integrated_z_var = np.sqrt(integrated_z_var / time_scaling)
        else:
            integrated_z_var = -9999.0
        z_var.append(integrated_z_var)
    return (np.array(z_mean), np.array(z_var))

def compute_mean_var_from_2D_array_sum(test_swot_z_obs, reach_t):
    z_mean = []
    check_na_vector = np.vectorize(calc.check_na)
    for n in range(0, test_swot_z_obs.shape[0]):
        integrated_z_mean = 0.0
        idx_sort = test_swot_z_obs[n, :].argsort()
        sorted_array = test_swot_z_obs[n, :][idx_sort]
        sorted_array_mask = check_na_vector(sorted_array)
        sorted_array = sorted_array[~sorted_array_mask]
        time_range = int(sorted_array.shape[0] / 3)
        if time_range > 0:
            for t in range(0, time_range):
                if calc.check_na(test_swot_z_obs[n, t]):
                    integrated_z_mean += 0.0
                else:
                    integrated_z_mean += test_swot_z_obs[n, t]
            integrated_z_mean = integrated_z_mean / time_range
        else:
            integrated_z_mean = np.nan
        z_mean.append(integrated_z_mean)
    z_var = []
    for n in range(0, test_swot_z_obs.shape[0]):
        integrated_z_var = 0.0
        idx_sort = test_swot_z_obs[n, :].argsort()
        sorted_array = test_swot_z_obs[n, :][idx_sort]
        sorted_array_mask = check_na_vector(sorted_array)
        sorted_array = sorted_array[~sorted_array_mask]
        time_range = int(sorted_array.shape[0] / 3)
        if time_range > 0.0:
            for t in range(0, time_range):
                if calc.check_na(test_swot_z_obs[n, t]):
                    integrated_z_var += 0.0
                else:
                    integrated_z_var += (test_swot_z_obs[n, t] - z_mean[n]) ** 2
                integrated_z_var = np.sqrt(integrated_z_var / time_range)
        else:
            integrated_z_var = np.nan
        z_var.append(integrated_z_var)
    return (np.array(z_mean), np.array(z_var))

def grad_variance(x):
    deriv_x = []
    for n in range(1, len(x)):
        deriv_x.append((x[n] - x[n - 1]) / 200.0)
    return np.array(deriv_x)

def get_weighted_q_data(dates_swot, q_mean):
    tot_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    indices = np.where(np.array(dates_swot) != '')
    indices = indices[0]
    first_date = dates_swot[indices[0]]
    last_date = dates_swot[indices[-1]]
    if first_date == '' and last_date == '':
        return -1
    i = int(first_date[5:7]) - 1
    j = int(last_date[5:7]) - 1
    new_dates_swot = np.array(dates_swot)[indices]
    dates = [datetime.strptime(date, '%Y-%m-%d') for date in new_dates_swot]
    days_in_months = {}
    for l in range(len(dates) - 1):
        days_in_months = count_days_in_months(dates[l], dates[l + 1], days_in_months)
    days_tot = np.full((1, 12), 0)[0]
    for (year, month), days in sorted(days_in_months.items()):
        days_tot[month - 1] = days
        logging.debug(f'{year}-{month:02d}: {days} days')
    if i == j and first_date[0:4] == last_date[0:4]:
        Qwbm = q_mean[i]
    else:
        Qwbm = 0
        for k in range(0, 12):
            mod = k % 11
            Qwbm += days_tot[mod] * q_mean[mod]
        time_start = date_to_seconds(first_date)
        time_end = date_to_seconds(last_date)
        delta_t = (time_end - time_start) / 86400
        Qwbm = Qwbm / delta_t
    return Qwbm

def interp_pdf_tables(n, u, x, y):
    istop = 0
    i = 0
    while i < n and istop == 0:
        i = i + 1
        if x[i] > u:
            x1 = x[i - 1]
            x2 = x[i]
            v = y[i - 1] + (y[i] - y[i - 1]) * (u - x1) / (x2 - x1)
            istop = 1
    if u <= x[0]:
        v = y[0]
    if u >= x[n]:
        v = y[n]
    return v

def disable_prints():
    sys.stdout = open(os.devnull, 'w')

def enable_prints():
    sys.stdout = sys.__stdout__

def get_reach_dataset(param_dict, index=None):
    """Extract and return data associated with a reach identifier from JSON file
    and AWS_BATCH_JOB_ARRAY_INDEX.
    Parameters
    ----------
    param_dict : dict
    Returns
    -------
    dict
    Reach dataset
    """
    reachjson = param_dict['json_path']
    if param_dict['aws']:
        if os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX'):
            index = int(os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX'))
            param_dict['index'] = index
    with open(reachjson) as jsonfile:
        data = json.load(jsonfile)
    if index != None and param_dict['run_type'] == 'seq':
        out_data = data[index]
    elif index != None and param_dict['run_type'] == 'set':
        if isinstance(data[index], dict):
            out_data = [data[index]]
        else:
            out_data = data[index]
    else:
        out_data = data
    return out_data

def write_to_log(file_path, string):
    with open(file_path, 'a') as file:
        file.write(string + '\n')

def get_svs_data(svs_dataset, reach_id):
    station_q = None
    station_data = None
    station_qt = None
    if svs_dataset != None:
        svs_rids_v16 = get_nc_variable_data(svs_dataset, 'reach_id_v16')
        svs_rids_v17 = get_nc_variable_data(svs_dataset, 'reach_id_v17')
        svs_Q = get_nc_variable_data(svs_dataset, 'Q')
        svs_t = get_nc_variable_data(svs_dataset, 'time')
        out_svs_date = daynum_to_date(svs_t, '2023-01-01')
        target_int = int(reach_id)
        target_str = str(reach_id)
        svs_match_rows = None
        if params.svs_reach_id_var == 'reach_id_v16':
            svs_match_rows_v16 = np.where(svs_rids_v16 == target_int)[0]
            svs_match_rows = svs_match_rows_v16
        elif params.svs_reach_id_var == 'reach_id_v17':
            svs_match_rows_v17_preliminary = np.where(svs_rids_v17 == target_str)[0]
            if np.array(svs_match_rows_v17_preliminary).size > 0:
                svs_match_rows_v17 = svs_match_rows_v17_preliminary
                svs_match_rows = svs_match_rows_v17
            else:
                logging.info('No easy matching for stations in v17, looping (slower).')
                for i in range(len(svs_rids_v17)):
                    if svs_dataset['multi_match_v17'][i] != 0:
                        for j in range(1, len(svs_rids_v17[i])):
                            if svs_rids_v17[i][j] == target_str:
                                svs_match_rows = [i]
                                print('Found match in multi_match_v17 at row', i, 'and index', j)
                                break
                if not svs_match_rows:
                    logging.info('No match found in multi_match_v17 either.')
                    svs_match_rows = []
        if np.array(svs_match_rows).size > 0:
            logging.info(f'Match found in SVS for reach {reach_id} at row index {svs_match_rows[0]}.')
            svs_row_idx = int(svs_match_rows[0])
            index_station = int(svs_dataset['station'][svs_row_idx]) - 1
            if np.array(index_station).size > 0:
                station_q = svs_Q[index_station, :]
                station_date = out_svs_date
                calibration_value = 0
                epoch = datetime(1, 1, 1)
                days_since_0001 = [(dt - epoch).total_seconds() / 86400 for dt in station_date]
                station_qt = days_since_0001
        else:
            logging.info(f'No match found in SVS for reach {reach_id}.')
        svs_dataset.close()
    return (station_q, station_date, station_qt)

def get_external_gauge_data(csv_file, reach_id):
    station_q = None
    station_date = None
    station_qt = None
    if csv_file == '':
        logging.warning('External gauge file path is empty.')
        return (station_q, station_date, station_qt)
    if not Path(csv_file).is_file():
        logging.warning(f'External gauge file not found: {csv_file}')
        return (station_q, station_date, station_qt)
    try:
        gauge_df = pd.read_csv(csv_file)
    except Exception as exc:
        logging.warning(f"Failed to read external gauge file '{csv_file}': {exc}")
        return (station_q, station_date, station_qt)
    required_columns = ['v17_reach_id', 'measurement_dateTime', 'Q_m3s']
    missing_columns = [col for col in required_columns if col not in gauge_df.columns]
    if len(missing_columns) > 0:
        logging.warning(f"Missing columns in external gauge file '{csv_file}': {missing_columns}. Expected columns are ['v17_reach_id', 'measurement_dateTime', 'Q_m3s'].")
        return (station_q, station_date, station_qt)
    try:
        gauge_df['v17_reach_id'] = pd.to_numeric(gauge_df['v17_reach_id'], errors='coerce')
        current_reach_id = int(reach_id)
    except Exception as exc:
        logging.warning(f"Could not cast reach_id '{reach_id}' to int for external gauge file filtering: {exc}")
        return (station_q, station_date, station_qt)
    filtered_df = gauge_df[gauge_df['v17_reach_id'] == current_reach_id].copy()
    if filtered_df.empty:
        logging.info(f'No external gauge data found for reach {current_reach_id}.')
        return (station_q, station_date, station_qt)
    filtered_df['measurement_dateTime'] = pd.to_datetime(filtered_df['measurement_dateTime'], utc=True, errors='coerce')
    filtered_df['Q_m3s'] = pd.to_numeric(filtered_df['Q_m3s'], errors='coerce')
    filtered_df = filtered_df.dropna(subset=['measurement_dateTime', 'Q_m3s'])
    if filtered_df.empty:
        logging.info(f'External gauge data exists for reach {current_reach_id} but no valid datetime/Q rows remain after cleaning.')
        return (station_q, station_date, station_qt)
    filtered_df = filtered_df.sort_values(by='measurement_dateTime')
    station_q = np.ma.masked_invalid(filtered_df['Q_m3s'].to_numpy(dtype=float))
    datetime_array = [ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts for ts in filtered_df['measurement_dateTime'].dt.tz_convert(None).tolist()]
    station_date = np.array([dt.strftime('%Y-%m-%d') for dt in datetime_array])
    epoch = datetime(1, 1, 1)
    station_qt = np.array([(dt - epoch).total_seconds() / 86400 for dt in datetime_array], dtype=float)
    return (station_q, station_date, station_qt)

def get_external_friction_data(csv_file, reach_id):
    friction_values = None
    friction_dates = None
    friction_t = None
    if csv_file == '':
        logging.warning('External friction file path is empty.')
        return (friction_values, friction_dates, friction_t)
    if not Path(csv_file).is_file():
        logging.warning(f'External friction file not found: {csv_file}')
        return (friction_values, friction_dates, friction_t)
    try:
        friction_df = pd.read_csv(csv_file)
    except Exception as exc:
        logging.warning(f"Failed to read external friction file '{csv_file}': {exc}")
        return (friction_values, friction_dates, friction_t)
    reach_col = 'reach_id_v17' if 'reach_id_v17' in friction_df.columns else 'v17_reach_id' if 'v17_reach_id' in friction_df.columns else None
    date_candidates = ['measurement_dateTime', 'date', 'datetime', 'Date', 'DATE']
    date_col = next((col for col in date_candidates if col in friction_df.columns), None)
    manning_col = 'Manning_n' if 'Manning_n' in friction_df.columns else None
    missing_columns = []
    if reach_col is None:
        missing_columns.append('reach_id_v17')
    if manning_col is None:
        missing_columns.append('Manning_n')
    if date_col is None:
        missing_columns.append('date')
    if len(missing_columns) > 0:
        logging.warning(f"Missing columns in external friction file '{csv_file}': {missing_columns}. Expected at least reach id, date, and Manning_n columns.")
        return (friction_values, friction_dates, friction_t)
    try:
        friction_df[reach_col] = pd.to_numeric(friction_df[reach_col], errors='coerce')
        current_reach_id = int(reach_id)
    except Exception as exc:
        logging.warning(f"Could not cast reach_id '{reach_id}' to int for external friction filtering: {exc}")
        return (friction_values, friction_dates, friction_t)
    filtered_df = friction_df[friction_df[reach_col] == current_reach_id].copy()
    if filtered_df.empty:
        logging.info(f'No external friction data found for reach {current_reach_id}.')
        return (friction_values, friction_dates, friction_t)
    filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], utc=True, errors='coerce')
    filtered_df[manning_col] = pd.to_numeric(filtered_df[manning_col], errors='coerce')
    filtered_df = filtered_df.dropna(subset=[date_col, manning_col])
    if filtered_df.empty:
        logging.info(f'External friction data exists for reach {current_reach_id} but no valid datetime/Manning_n rows remain after cleaning.')
        return (friction_values, friction_dates, friction_t)
    filtered_df = filtered_df.sort_values(by=date_col)
    datetime_array = [ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts for ts in filtered_df[date_col].dt.tz_convert(None).tolist()]
    epoch_2000 = datetime(2000, 1, 1)
    friction_t = np.array([(dt - epoch_2000).total_seconds() for dt in datetime_array], dtype=float)
    friction_dates = np.array([dt.strftime('%Y-%m-%d') for dt in datetime_array])
    friction_values = np.ma.masked_invalid(filtered_df[manning_col].to_numpy(dtype=float))
    return (friction_values, friction_dates, friction_t)

def get_sos_data(sos_file, reach_id, reach_t, param_dict):
    sos_dict = {}
    sos_dataset = Dataset(sos_file)
    if params.auto_spread_for_runs and (not params.use_dynamic_spread):
        if 'unconstrained' in sos_dataset.getncattr('run_type'):
            params.shape03 = 10.0
        elif sos_dataset.getncattr('run_type') == 'constrained':
            params.shape03 = 30.0
    sos_rids = get_nc_variable_data(sos_dataset, 'reaches/reach_id')
    index = np.where(sos_rids == reach_id)
    if np.array(index[0]).size > 0:
        sos_dict['q_monthly_mean'] = get_nc_variable_data(sos_dataset, 'model/monthly_q')[index, :][0][0]
        sos_dict['reach_qwbm'] = get_nc_variable_data(sos_dataset, 'model/mean_q')[index]
        sos_dict['quantiles'] = sos_dataset['model']['flow_duration_q'][index[0], :][0]
        sos_dict['quant_mean'], sos_dict['quant_var'] = calc.compute_mean_discharge_from_SoS_quantiles(sos_dict['quantiles'])
        sos_dict['flow_duration_q'] = sos_dataset['model']['flow_duration_q'][index[0], :][0]
    else:
        logging.warning(f'Reach {reach_id} not found in SoS dataset.')
        masked_data = np.ma.masked_values(np.array([np.nan]), value=-9999.0)
        sos_dict['q_monthly_mean'] = masked_data
        sos_dict['reach_qwbm'] = masked_data
        sos_dict['quantiles'] = masked_data
        sos_dict['quant_mean'], sos_dict['quant_var'] = (np.nan, np.nan)
        sos_dict['flow_duration_q'] = masked_data
    if sos_dict['quantiles'].mask.all():
        params.use_mean_for_bounds = True
    else:
        params.use_mean_for_bounds = False
    if params.qsdev_activate or params.q_mean_computed:
        masked_data = np.ma.masked_values(np.array([float(sos_dict['quant_mean'])]), value=-9999.0)
        sos_dict['reach_qwbm'] = deepcopy(masked_data)
    if params.use_flow_duration_q:
        sos_dict['quant_mean'], sos_dict['quant_var'], sos_dict['quant_cv'] = compute_cv_from_flow_duration_curve(sos_dict['flow_duration_q'])
        sos_dict['flow_duration_mean'] = sos_dict['quant_mean']
        sos_dict['flow_duration_var'] = sos_dict['quant_var']
    if params.qsdev_activate:
        if params.qsdev_option == 0:
            masked_data2 = np.ma.masked_values(np.array([float(sos_dict['quant_var'])]), value=-9999.0)
        elif params.qsdev_option == 1:
            masked_data2 = np.ma.masked_values(np.array([float(sos_dict['quant_var'] / sos_dict['quant_mean'])]), value=-9999.0)
        sos_dict['reach_qsdev'] = masked_data2
    elif not params.qsdev_activate:
        sos_dict['reach_qsdev'] = 0.0
    if calc.check_na(sos_dict['reach_qwbm']):
        pass
    if param_dict['override_q_prior']:
        masked_data = np.ma.masked_values(np.array([float(param_dict['q_prior_value'])]), value=-9999.0)
        sos_dict['reach_qwbm'] = masked_data
    sos_reach_id = get_nc_variable_data(sos_dataset, 'reaches/reach_id')
    index_reach_sos = np.array(np.where(sos_reach_id == int(reach_id)))[0]
    sos_dict['station_q'], sos_dict['station_date'], sos_dict['station_qt'], _ = get_station_q_and_qt(sos_dataset, reach_id)
    if params.use_SVS:
        station_q = None
        station_data = None
        station_qt = None
        if params.SVS_file != '':
            svs_dataset = Dataset(params.SVS_file)
        else:
            svs_dataset = None
        station_q, station_dates, station_qt = get_svs_data(svs_dataset, reach_id)
        if station_q is not None and station_dates is not None and (station_qt is not None):
            sos_dict['station_q'] = station_q
            sos_dict['station_date'] = station_dates
            sos_dict['station_qt'] = station_qt
            logging.info(f'Using SVS data for reach {reach_id}.')
    if params.use_gauge_external_file:
        station_q = None
        station_date = None
        station_qt = None
        station_q, station_date, station_qt = get_external_gauge_data(params.gauge_external_file, reach_id)
        if station_q is not None and station_date is not None and (station_qt is not None):
            sos_dict['station_q'] = station_q
            sos_dict['station_date'] = station_date
            sos_dict['station_qt'] = station_qt
            logging.info(f'Using external gauge CSV data for reach {reach_id}.')
        else:
            sos_dict['station_q'] = []
    if np.array(sos_dict['station_q']).size == 0:
        logging.warning(f'No station in SoS file is available for {reach_id}.')
        masked_data = np.ma.masked_values(np.array([np.nan]), value=-9999.0)
        sos_dict['q_mean_station'] = masked_data
        masked_data2 = np.ma.masked_values(np.array([np.nan]), value=-9999.0)
        sos_dict['q_std_station'] = masked_data2
        if params.reference_force_quit:
            logging.debug('Stations force quitting 1.')
            sos_dataset.close()
            return (0, {})
    filtered_station_df = pd.DataFrame()
    if np.array(sos_dict['station_q']).size > 0 and sos_dict['station_q'].mask.all() != True:
        ref = np.datetime64('2000-01-01')
        swot_datetimes = ref + reach_t * np.timedelta64(1, 's')
        datetimes_np = np.array(sos_dict['station_date'], dtype='datetime64[ns]')
        ref = np.datetime64('2000-01-01')
        sos_dict['station_time_in_days'] = (datetimes_np - ref) / np.timedelta64(1, 'D')
        q_mean, q_std, filtered_station_df = use_stations_for_q_prior(swot_datetimes, sos_dict['station_q'], sos_dict['station_date'], sos_dict['station_time_in_days'])
        sos_dict['filtered_station_df'] = filtered_station_df
        if filtered_station_df.empty:
            logging.warning(f'No valid station data after filtering for reach {reach_id}.')
            masked_data = np.ma.masked_values(np.array([np.nan]), value=-9999.0)
            sos_dict['q_mean_station'] = masked_data
            masked_data2 = np.ma.masked_values(np.array([np.nan]), value=-9999.0)
            sos_dict['q_std_station'] = masked_data2
            if params.reference_force_quit:
                logging.debug('Stations force quitting 2.')
                sos_dataset.close()
                return (0, {})
        else:
            sos_dict['station_qt_2000'] = filtered_station_df['source_times_in_days'] / 86400
            sos_dict['station_df_used'] = filtered_station_df
            masked_data = np.ma.masked_values(np.array([q_mean]), value=-9999.0)
            sos_dict['q_mean_station'] = masked_data
            masked_data2 = np.ma.masked_values(np.array([float(q_std)]), value=-9999.0)
            sos_dict['q_std_station'] = masked_data2
        if param_dict['q_prior_from_stations']:
            sos_dict['reach_qwbm'] = deepcopy(sos_dict['q_mean_station'])
            if params.qsdev_activate:
                if params.qsdev_option == 0:
                    masked_data2 = np.ma.masked_values(np.array([float(sos_dict['q_std_station'])]), value=-9999.0)
                elif params.qsdev_option == 1:
                    masked_data2 = np.ma.masked_values(np.array([float(sos_dict['q_std_station'] / sos_dict['q_mean_station'])]), value=-9999.0)
                sos_dict['reach_qsdev'] = masked_data2
            if q_mean <= 1.0:
                logging.warning(call_error_message(506))
                return (0, {})
    if params.q_prior_from_ML:
        logging.info('Using ML priors.')
        ml_ds = Dataset(params.ML_time_series_path)
        index_reach = np.where(reach_id == ml_ds['reach_id'][:])[0]
        if np.array(index_reach).size > 0.0:
            flow = ml_ds['flow'][index_reach, :][0]
            time = ml_ds['time'][index_reach, :][0]
            flow = np.ma.array(flow)
            time = np.ma.array(time)
            mask = ~flow.mask & ~time.mask
            valid_flow = flow[mask]
            valid_times = time[mask]
            if valid_flow.size > 0:
                if valid_flow.ndim > 1:
                    valid_flow = valid_flow[0]
            if valid_times.size > 0:
                if valid_times.ndim > 1:
                    valid_times = valid_times[0]
            sorted_indices = np.argsort(valid_times)
            valid_times = valid_times[sorted_indices]
            valid_flow = valid_flow[sorted_indices]
            valid_times_days = valid_times / 86400
            if valid_flow.size > 0 and valid_flow.mask.all() != True:
                from lib.lib_dates import daynum_to_date
                ref = np.datetime64('2000-01-01')
                swot_datetimes = ref + reach_t * np.timedelta64(1, 's')
                ml_days = np.array(valid_times_days)
                ref = np.datetime64('2000-01-01')
                datetimes_np = ref + ml_days * np.timedelta64(1, 'D')
                q_mean, q_std, filtered_ml_df = use_stations_for_q_prior(swot_datetimes, valid_flow, datetimes_np, ml_days)
                sos_dict['ML_qt_2000'] = filtered_ml_df['source_times_in_days']
                sos_dict['ML_df'] = filtered_ml_df
                masked_data = np.ma.masked_values(np.array([q_mean]), value=-9999.0)
                sos_dict['q_mean_ML'] = masked_data
                sos_dict['reach_qwbm'] = deepcopy(sos_dict['q_mean_ML'])
                sos_dict['q_std_ML'] = q_std
                if params.ml_stations_sync:
                    if not filtered_station_df.empty:
                        station_t = np.ma.masked_values(np.array(filtered_station_df['station_qt_2000_days']), value=-9999.0)
                        q_mean2, q_std2, filtered_df_out = use_stations_for_q_prior(station_t * 86400, valid_flow, out_ml_date, reach_id, valid_times_days, 'ML', param_dict, option2=2, additional_t=reach_t)
                        sos_dict['ML_qt_2000'] = filtered_df_out['ML_qt_2000']
                        sos_dict['ML_df'] = filtered_df_out
                        masked_data = np.ma.masked_values(np.array([q_mean2]), value=-9999.0)
                        sos_dict['q_mean_ML'] = masked_data
                        if params.qsdev_activate:
                            if params.qsdev_option == 0:
                                masked_data2 = np.ma.masked_values(np.array([float(q_std2)]), value=-9999.0)
                            elif params.qsdev_option == 1:
                                masked_data2 = np.ma.masked_values(np.array([float(q_std2 / q_mean2)]), value=-9999.0)
                        else:
                            masked_data2 = np.ma.masked_values(np.array([float(q_std2)]), value=-9999.0)
                        sos_dict['q_std_ML'] = masked_data2
                        sos_dict['reach_qwbm'] = deepcopy(sos_dict['q_mean_ML'])
                    elif params.ML_priors_force_quit:
                        logging.warning('ML force quitting.')
                        return (0, {})
                if params.qsdev_activate:
                    if params.qsdev_option == 0:
                        masked_data2 = np.ma.masked_values(np.array([float(sos_dict['q_std_ML'])]), value=-9999.0)
                    elif params.qsdev_option == 1:
                        masked_data2 = np.ma.masked_values(np.array([float(sos_dict['q_std_ML'] / sos_dict['q_mean_ML'])]), value=-9999.0)
                    sos_dict['reach_qsdev'] = masked_data2
                if q_mean <= 1.0:
                    logging.warning(call_error_message(506))
                    return (0, {})
        else:
            pass
            logging.warning('Reach has no ML priors.')
            if params.ML_priors_force_quit:
                logging.warning('ML force quitting.')
                return (0, {})
    if sos_dict['reach_qwbm'] <= 1 and (not params.activate_facc):
        sos_dict['reach_qwbm'] = np.ma.masked_values(np.nan, value=-9999.0)
        logging.warning('Replacing QWBM with NaN for reach {} because QWBM <= 1 and FACC is not activated.'.format(reach_id))
    sos_dataset.close()
    return sos_dict

def compute_cv_from_flow_duration_curve(flow_duration_q):
    nq = len(flow_duration_q)
    bin = 1.0 / float(nq)
    quant = np.zeros(nq)
    dquant = np.zeros(nq + 1)
    for i in range(0, len(flow_duration_q)):
        quant[i] = flow_duration_q[i]
    dquant[0] = (3.0 * quant[0] - quant[1]) / 2.0
    for i in range(0, nq - 1):
        dquant[i + 1] = (quant[i] + quant[i + 1]) / 2.0
    dquant[nq] = (3.0 * quant[nq - 1] - quant[nq - 2]) / 2.0
    for i in range(0, nq):
        quant[i] = dquant[nq - i]
    dquant_max = 0.0
    for i in range(0, nq - 1):
        dquant[i] = bin / (quant[i + 1] - quant[i]) * (quant[i + 1] - quant[i])
        if dquant[i] > dquant_max:
            dquant_max = dquant[i]
    ss = 0.0
    ss1 = 0.0
    for i in range(0, nq):
        ss = ss + quant[i] * bin
        ss1 = ss1 + bin
    quant_mean0 = ss / ss1
    ss = 0.0
    ss1 = 0.0
    for i in range(0, nq - 1):
        ss = ss + (quant[i + 1] + quant[i]) / 2.0 * dquant[i]
        ss1 = ss1 + dquant[i]
    quant_mean = ss / ss1
    ss = 0.0
    ss1 = 0.0
    for i in range(0, nq - 1):
        ss = ss + ((quant[i + 1] + quant[i]) / 2.0 - quant_mean) ** 2 * dquant[i]
        ss1 = ss1 + dquant[i]
    quant_var = np.sqrt(ss / ss1)
    Q_MEAN_GRD = quant_mean
    Q_STD_GRD = quant_var
    quant_CV = quant_var / quant_mean
    return (Q_MEAN_GRD, Q_STD_GRD, quant_CV)

def detect_suspicious_node_positions_from_sword(node_ids, dist_out=None, node_length=None, zscore_threshold=8.0):
    node_ids = np.asarray(node_ids)
    if node_ids.size < 4:
        return {'suspicious_node_ids': np.array([], dtype=node_ids.dtype if node_ids.size > 0 else int), 'suspicious_node_indexes': np.array([], dtype=int), 'segment_distances_m': np.array([], dtype=float), 'distance_threshold_m': np.nan, 'distance_source': 'none'}
    segment_distances = None
    distance_source = 'none'
    if node_length is not None:
        node_length = np.asarray(node_length, dtype=float)
        if node_length.size >= 2:
            segment_distances = np.abs(node_length[:-1])
            distance_source = 'node_length'
    if segment_distances is None and dist_out is not None:
        dist_out = np.asarray(dist_out, dtype=float)
        if dist_out.size >= 2:
            segment_distances = np.abs(np.diff(dist_out))
            distance_source = 'dist_out'
    if segment_distances is None:
        return {'suspicious_node_ids': np.array([], dtype=node_ids.dtype), 'suspicious_node_indexes': np.array([], dtype=int), 'segment_distances_m': np.array([], dtype=float), 'distance_threshold_m': np.nan, 'distance_source': 'none'}
    if np.all(~np.isfinite(segment_distances)):
        return {'suspicious_node_ids': np.array([], dtype=node_ids.dtype), 'suspicious_node_indexes': np.array([], dtype=int), 'segment_distances_m': segment_distances, 'distance_threshold_m': np.nan, 'distance_source': distance_source}
    med = np.nanmedian(segment_distances)
    mad = np.nanmedian(np.abs(segment_distances - med))
    if not np.isfinite(mad) or mad < 1e-09:
        threshold = np.nanpercentile(segment_distances, 99)
    else:
        robust_sigma = 1.4826 * mad
        threshold = med + zscore_threshold * robust_sigma
    suspicious_edges = np.where(segment_distances > threshold)[0]
    if suspicious_edges.size == 0:
        suspicious_node_indexes = np.array([], dtype=int)
    else:
        suspicious_node_indexes = np.unique(np.concatenate([suspicious_edges, suspicious_edges + 1]))
    return {'suspicious_node_ids': node_ids[suspicious_node_indexes] if suspicious_node_indexes.size > 0 else np.array([], dtype=node_ids.dtype), 'suspicious_node_indexes': suspicious_node_indexes, 'segment_distances_m': segment_distances, 'distance_threshold_m': float(threshold) if np.isfinite(threshold) else np.nan, 'distance_source': distance_source}

def get_sword_data(sword_file, reach_id):
    sword_dict = {}
    sword_flag_dict = {}
    sword_dataset = Dataset(sword_file)
    index_node_ids = np.array(np.where(get_nc_variable_data(sword_dataset, 'nodes/reach_id') == int(reach_id)))[0]
    sword_nodes_id = sword_dataset['nodes']['node_id'][index_node_ids]
    sword_node_order = None
    if 'node_order' in sword_dataset['nodes'].variables:
        sword_node_order = sword_dataset['nodes']['node_order'][index_node_ids]
    sword_dict['node_order'] = sword_node_order
    logging.info(f'Reach {reach_id}: SWORD node order before reordering: {sword_node_order}')
    sword_nodes_id_reordered_index, sword_nodes_id_reordered = reorder_ids_with_indices(sword_nodes_id, sword_node_order=sword_node_order, params=params)
    index_node_ids = index_node_ids[sword_nodes_id_reordered_index]
    if not params.opt_sword_boost:
        sword_dict['node_length'] = get_nc_variable_data(sword_dataset, 'nodes/node_length')[index_node_ids]
        sword_dict['dist_out'] = get_nc_variable_data(sword_dataset, 'nodes/dist_out')[index_node_ids]
        if params.pankaj_test:
            middle = round(len(index_node_ids) / 2)
            node = index_node_ids[middle]
            sword_dict['dist_out'] = get_nc_variable_data(sword_dataset, 'nodes/dist_out')[node]
            sword_dict['node_length'] = get_nc_variable_data(sword_dataset, 'nodes/node_length')[node]
    elif params.opt_sword_boost:
        ref_index_rch = np.array(np.where(get_nc_variable_data(sword_dataset, 'reaches/reach_id') == int(reach_id)))[0]
        index_rch_node = np.load('index_rch_node.npy')
        i1 = int(index_rch_node[ref_index_rch, 0])
        i2 = int(index_rch_node[ref_index_rch, 1])
        list_nodes = np.arange(i1, i2 + 1)
        sword_dict['node_length'] = get_nc_variable_data(sword_dataset, 'nodes/node_length')[list_nodes]
        sword_dict['dist_out'] = get_nc_variable_data(sword_dataset, 'nodes/dist_out')[list_nodes]
    sword_dict['node_w_mean'] = get_nc_variable_data(sword_dataset, 'nodes/width')[index_node_ids]
    index_reach_sword = np.array(np.where(get_nc_variable_data(sword_dataset, 'reaches/reach_id') == int(reach_id)))[0]
    sword_dict['reach_w_mean'] = get_nc_variable_data(sword_dataset, 'reaches/width')[index_reach_sword]
    sword_dict['reach_slope'] = get_nc_variable_data(sword_dataset, 'reaches/slope')[index_reach_sword]
    sword_dict['reach_length'] = get_nc_variable_data(sword_dataset, 'reaches/reach_length')[index_reach_sword]
    sword_flag_dict = {'reach': {'reach_width_min': sword_dict['reach_w_mean'], 'reach_slope_min': sword_dict['reach_slope'], 'reach_length_min': sword_dict['reach_length']}, 'node': {'reach_length_min': sword_dict['node_w_mean']}}
    sword_dict['facc'] = get_nc_variable_data(sword_dataset, 'reaches/facc')[index_reach_sword]
    sword_dataset.close()
    return (sword_dict, sword_flag_dict)

def get_swot_data(swot_file, param_dict, sword_dict):
    swot_dict = {}
    flag_dict = {}
    swot_dataset = Dataset(swot_file)
    try:
        swot_observations = get_nc_variable_data(swot_dataset, f'observations')
        swot_dict['pass_ids'] = np.array([b''.join(row.compressed()).decode('utf-8') for row in swot_observations])
    except Exception as e:
        logging.error(f'Error reading pass_ids from SWOT dataset: {e}')
        swot_dict['pass_ids'] = np.array([])
    swot_node_ids = get_nc_variable_data(swot_dataset, 'node/node_id')
    if np.array(swot_node_ids).size == 0:
        logging.error('SWOT file has no node_id field !')
        return ({}, {})
    swot_nodes_id_reordered_index, swot_nodes_id_reordered = reorder_ids_with_indices(swot_node_ids, sword_node_order=sword_dict['node_order'], params=params)
    swot_node_ids = swot_node_ids[swot_nodes_id_reordered_index]
    check_na_vector = np.vectorize(calc.check_na)
    reach_param_flags = ['obs_frac_n', 'dark_frac', 'xovr_cal_q', 'reach_q_b', 'slope_r_u', 'partial_f', 'xtrk_dist', 'ice_clim_f']
    flag_dict['reach'] = {}
    for attribute in reach_param_flags:
        if get_nc_variable_data(swot_dataset, f'reach/{attribute}').shape != (0,):
            flag_dict['reach'][attribute] = get_nc_variable_data(swot_dataset, f'reach/{attribute}')
    node_param_flags = ['xtrk_dist', 'dark_frac', 'wse_u', 'width_u', 'node_q', 'node_q_b', 'xovr_cal_q', 'wse_r_u', 'n_good_pix']
    flag_dict['node'] = {}
    for attribute in node_param_flags:
        if get_nc_variable_data(swot_dataset, f'node/{attribute}').shape != (0,):
            flag_dict['node'][attribute] = get_nc_variable_data(swot_dataset, f'node/{attribute}')[swot_nodes_id_reordered_index]
    flag_dict['nx'] = get_nc_variable_data(swot_dataset, 'node/wse').shape[0]
    flag_dict['nt'] = get_nc_variable_data(swot_dataset, 'node/wse').shape[1]
    logging.info(f'Loading file {swot_file} - nx : {flag_dict['nx']} nt {flag_dict['nt']}')
    if hasattr(swot_dataset['reach/wse'], 'valid_min'):
        swot_dict['valid_min_z'] = swot_dataset['reach/wse'].valid_min
    else:
        swot_dict['valid_min_z'] = params.valid_min_z
    if 'reach/d_x_area' in swot_dataset.variables and hasattr(swot_dataset['reach/d_x_area'], 'valid_min'):
        swot_dict['valid_min_dA'] = swot_dataset['reach/d_x_area'].valid_min
    else:
        swot_dict['valid_min_dA'] = params.valid_min_dA
    swot_dict['reach_s'] = get_nc_variable_data(swot_dataset, 'reach/slope2')
    if 'reach/d_x_area' in swot_dataset.variables:
        swot_dict['reach_dA'] = get_nc_variable_data(swot_dataset, 'reach/d_x_area')
    else:
        swot_dict['reach_dA'] = np.full(swot_dict['reach_s'].shape, np.nan)
    swot_dict['reach_s'] = get_nc_variable_data(swot_dataset, 'reach/slope2')
    swot_dict['reach_w'] = get_nc_variable_data(swot_dataset, 'reach/width')
    swot_dict['reach_z'] = get_nc_variable_data(swot_dataset, 'reach/wse')
    swot_dict['reach_t'] = get_nc_variable_data(swot_dataset, 'reach/time')
    swot_dict['node_w'] = get_nc_variable_data(swot_dataset, 'node/width')[swot_nodes_id_reordered_index]
    swot_dict['node_z'] = get_nc_variable_data(swot_dataset, 'node/wse')[swot_nodes_id_reordered_index]
    if 'reach_dA' in swot_dict.keys():
        swot_dict['node_dA'] = np.ones(swot_dict['node_z'].shape) * swot_dict['reach_dA']
    swot_dict['node_s'] = np.ones(swot_dict['node_z'].shape) * swot_dict['reach_s']
    swot_dict['node_t'] = get_nc_variable_data(swot_dataset, 'node/time')[swot_nodes_id_reordered_index]
    swot_dict['node_id'] = swot_nodes_id_reordered
    if swot_dict['reach_s'].mask.all():
        if not swot_dict['node_s'][0].mask.all():
            swot_dict['reach_s'] = swot_dict['node_s'][0]
    swot_dataset.close()
    return (swot_dict, flag_dict)

def get_input_data(param_dict, reach_dict):
    sword_file = param_dict['sword_dir'].joinpath(reach_dict['sword'])
    swot_file = param_dict['swot_dir'].joinpath(reach_dict['swot'])
    sos_file = param_dict['sos_dir'].joinpath(reach_dict['sos'])
    data_dict = {'reach_id': reach_dict['reach_id']}
    flag_dict = {'node': {}, 'reach': {}}
    if sword_file.exists():
        sword_dict, sword_flag_dict = get_sword_data(sword_file, reach_dict['reach_id'])
        flag_dict['node'].update(sword_flag_dict['node'])
        flag_dict['reach'].update(sword_flag_dict['reach'])
        data_dict.update(sword_dict)
    else:
        logging.error(call_error_message(504).format(reach_id=reach_dict['reach_id'], sword_file=sword_file))
        return (0, {})
    if swot_file.exists():
        swot_dict, flag_dict = get_swot_data(swot_file, param_dict, sword_dict)
        if 'node' in flag_dict:
            flag_dict['node'].update(sword_flag_dict['node'])
        if 'reach' in flag_dict:
            flag_dict['reach'].update(sword_flag_dict['reach'])
        data_dict.update(swot_dict)
    else:
        logging.error(call_error_message(505).format(reach_id=reach_dict['reach_id'], swot_file=swot_file))
        if param_dict['run_type'] == 'seq':
            return (0, {})
        elif param_dict['run_type'] == 'set':
            pass
    if sos_file.exists():
        if 'reach_t' in data_dict:
            sos_dict_tmp = get_sos_data(sos_file, reach_dict['reach_id'], data_dict['reach_t'], param_dict)
        else:
            sos_dict_tmp = (0, {})
        if sos_dict_tmp != (0, {}):
            sos_dict = deepcopy(sos_dict_tmp)
        elif params.use_fallback_sos:
            logging.info('Using fallback SoS file')
            sos_file2 = Path(params.fallback_sos_dir).joinpath(reach_dict['sos'])
            sos_dict_tmp = get_sos_data(sos_file2, reach_dict['reach_id'], data_dict['reach_t'], param_dict)
            if sos_dict_tmp != (0, {}) and sos_dict_tmp['station_q'].size > 0:
                swot_dict = deepcopy(sos_dict_tmp)
                logging.info(f'station was replaced: {sos_dict_tmp['station_q']}')
            else:
                logging.error('Fallback SoS file also has no station data, quitting')
                return (0, {})
        else:
            logging.warning('use_fallback_sos set to False, not replacing')
            return (0, {})
        data_dict.update(sos_dict)
    else:
        logging.error(call_error_message(104).format(reach_id=reach_dict['reach_id'], sos_file=sos_file))
    return (data_dict, flag_dict)

def change_info_for_bathy(matrix, max_row_size, sword_node_ids, observed_rows):
    n_total = len(sword_node_ids)
    if max_row_size > 1:
        filled_data = [np.full(max_row_size, np.nan) for _ in range(n_total)]
    else:
        filled_data = np.full(n_total, np.nan)
    for i, obs_idx in enumerate(observed_rows):
        if obs_idx < len(filled_data):
            filled_data[obs_idx] = matrix[i]
    return filled_data[::-1]

def write_output(output_path, param_dict, reach_id, output_dict, reach_number=0, dim_t=0, algo5_results={}, bb=np.nan, reliability='', nb_pts_bathy_max=10):
    """Write data to NetCDF output file.

    Parameters
    ----------
    output_path : Path
        Path to output NetCDF file
    param_dict : dict
        Configuration parameters (run_type, gnuplot_saving, etc.)
    reach_id : int
        Reach ID
    output_dict : dict
        Output data dictionary with keys: q_algo5, q_algo31, node_id, node_z, node_w, time_steps, etc.
    reach_number : int, optional
        Reach number for "set" mode (default 0)
    dim_t : int, optional
        Dimension of time when no data is available (default 0)
    algo5_results : dict, optional
        Algo5 results with A0, n parameters (default {})
    bb : float, optional
        Bathymetry parameter (default np.nan)
    reliability : str, optional
        Reliability flag (default "")
    """
    Path(output_path.parent).mkdir(parents=True, exist_ok=True)
    out_nc = Dataset(output_path, 'w', format='NETCDF4')
    fill_value = -999999999999.0
    out_nc.createDimension('nt', output_dict['time'].shape[0])
    out_nc.createDimension('times', output_dict['time'].shape[0])
    out_nc.createDimension('nx', len(output_dict.get('node_id', [])))
    out_nc.createDimension('nodes', len(output_dict.get('node_id', [])))
    out_nc.createDimension('nb_pts', nb_pts_bathy_max)
    logging.info(f'Writing output NetCDF file at {output_path}')
    logging.debug(f'nt times dimension size: {output_dict['time'].shape[0]}')
    logging.debug(f'nx nodes dimension size: {len(output_dict.get('node_id', []))}')
    logging.debug(f'nb_pts dimension size: {nb_pts_bathy_max}')
    nt_var = out_nc.createVariable('nt', 'i4', ('nt',))
    nt_var.units = 'time'
    nt_var[:] = range(output_dict['time'].shape[0])
    nx_var = out_nc.createVariable('nx', 'i4', ('nx',))
    nx_var.units = 'node'
    nx_var.long_name = 'number of nodes'
    nx_var[:] = range(1, len(output_dict.get('node_id', [])) + 1)
    times_var = out_nc.createVariable('times', 'f8', ('times',), fill_value=fill_value)
    times_var.units = 'days since 1st of January 2000'
    if output_dict.get('time') is not None and output_dict['time'].size > 0:
        times_var[:] = output_dict['time']
    out_nc.reach_id = reach_id
    out_nc.valid = int(output_dict.get('valid', 0))
    out_nc.reliability = reliability
    out_nc.stopped_stage = output_dict.get('stopped_stage', '')
    write_algo5_params(out_nc, nc_dict={}, algo5_results=algo5_results, reach_number=reach_number, output_dict=output_dict, param_dict=param_dict, fill_value=fill_value)
    bb_var = out_nc.createVariable('bb', 'f8', fill_value=fill_value)
    bb_var[:] = bb
    q_algo31, q_algo5 = write_discharge(out_nc, output_dict, reach_number, param_dict, fill_value)
    if params.optional_outputs:
        if 'alph1' in output_dict.keys():
            alph1 = out_nc.createVariable('alph1', 'f8', fill_value=fill_value)
            alph1[:] = output_dict['alph1']
    Kmi_acc = out_nc.createVariable('K', 'f8', fill_value=fill_value)
    Kmi_acc.long_name = 'estimated friction coefficient'
    Kmi_acc.description = 'estimated friction coefficient (m^(1/3)/s)'
    Zb_acc = out_nc.createVariable('Zb_acc', 'f8', fill_value=fill_value)
    Zb_acc.long_name = 'estimated bed elevation'
    Zb_acc.description = 'estimated bed elevation (m)'
    if 'Kmi_acc' in output_dict.keys():
        Kmi_acc[:] = output_dict['Kmi_acc']
    else:
        Kmi_acc[:] = np.nan
    if 'Zb_acc' in output_dict.keys():
        Zb_acc[:] = output_dict['Zb_acc']
    else:
        Zb_acc[:] = np.nan
    logging.info(f'Kmi_acc: {Kmi_acc[:]}')
    logging.info(f'Zb_acc: {Zb_acc[:]}')
    mean_elevation_profile = out_nc.createVariable('mean_elevation_profile', 'f8', ('nx',), fill_value=fill_value)
    mean_elevation_profile.long_name = 'mean elevation profile'
    mean_elevation_profile.description = 'mean elevation profile (m)'
    if 'mean_elevation_profile' in output_dict.keys():
        mean_elevation_profile[:] = output_dict['mean_elevation_profile']
    else:
        node_size = nx = len(out_nc.dimensions['nx'])
        mean_elevation_profile[:] = np.ones(node_size) * np.nan
    logging.info(f'mean_elevation_profile: {mean_elevation_profile[:]}')
    nb_pts_bathy_max = 10
    out_nc.createDimension('nb_pts_reach', nb_pts_bathy_max)
    reach_xr = out_nc.createVariable('reach_xr', 'f8', ('nb_pts_reach',), fill_value=fill_value)
    reach_xr.long_name = 'reach width bathymetry'
    reach_xr.description = 'reach width bathymetry (m)'
    reach_yr = out_nc.createVariable('reach_yr', 'f8', ('nb_pts_reach',), fill_value=fill_value)
    reach_yr.long_name = 'reach elevation bathymetry'
    reach_yr.description = 'reach elevation bathymetry (m)'
    if 'reach_xr' in output_dict.keys() and 'reach_yr' in output_dict.keys():
        reach_xr_data = np.array(np.pad(output_dict['reach_xr'], (0, nb_pts_bathy_max - len(output_dict['reach_xr'])), constant_values=np.nan))
        reach_yr_data = np.array(np.pad(output_dict['reach_yr'], (0, nb_pts_bathy_max - len(output_dict['reach_yr'])), constant_values=np.nan))
        reach_xr[:] = reach_xr_data
        reach_yr[:] = reach_yr_data
    else:
        reach_xr[:] = np.full(nb_pts_bathy_max, np.nan)
        reach_yr[:] = np.full(nb_pts_bathy_max, np.nan)
        logging.info(f'reach_xr: {reach_xr[:]}')
        logging.info(f'reach_yr: {reach_yr[:]}')
    SLOPEM1_constant = out_nc.createVariable('SLOPEM1_constant', 'f8', fill_value=fill_value)
    SLOPEM1_constant.long_name = 'SLOPEM1 constant'
    SLOPEM1_constant.description = 'SLOPEM1 constant (m)'
    if 'SLOPEM1_constant' in output_dict.keys():
        SLOPEM1_constant[:] = output_dict['SLOPEM1_constant'][0]
    else:
        SLOPEM1_constant[:] = np.nan
    logging.info(f'SLOPEM1_constant: {SLOPEM1_constant[:]}')
    if 'correlation_monthly' in output_dict.keys():
        correlation_monthly = out_nc.createVariable('correlation_monthly', 'f8', fill_value=fill_value)
        correlation_monthly.long_name = 'correlation monthly'
        correlation_monthly.description = 'correlation monthly (-)'
        if output_dict['correlation_monthly'] is not None:
            correlation_monthly[:] = output_dict['correlation_monthly']
        else:
            correlation_monthly[:] = np.nan
    if 'correlation_dynamic_slope' in output_dict.keys():
        correlation_dynamic_slope = out_nc.createVariable('correlation_dynamic_slope', 'f8', fill_value=fill_value)
        correlation_dynamic_slope.long_name = 'correlation dynamic slope'
        correlation_dynamic_slope.description = 'correlation dynamic slope (-)'
        if output_dict['correlation_dynamic_slope'] is not None:
            correlation_dynamic_slope[:] = output_dict['correlation_dynamic_slope']
        else:
            correlation_dynamic_slope[:] = np.nan
    out_nc.createDimension('n_quantiles', 3)
    quantile_matrix = out_nc.createVariable('quantile_matrix', 'f8', ('nx', 'n_quantiles'), fill_value=fill_value)
    quantile_matrix.long_name = 'quantile matrix of elevation profiles'
    quantile_matrix.description = 'quantile matrix (m)'
    if 'quantile_matrix' in output_dict:
        quantile_matrix[:, :] = output_dict['quantile_matrix']
    else:
        quantile_matrix[:, :] = np.nan
    logging.info(f'Quantile_matrix: {quantile_matrix[:, :]}')
    prior_used = out_nc.createVariable('prior_used', 'f8', fill_value=fill_value)
    prior_used.long_name = 'mean prior used'
    prior_used.description = "mean prior used in sic4dvar's estimation of discharge"
    if 'reach_qwbm' in output_dict.keys():
        prior_used[:] = output_dict['reach_qwbm']
    else:
        prior_used[:] = np.nan
    logging.info(f'prior_used: {prior_used[:]}')
    stages_not_saving = ['init', 'input_data_check']
    if output_dict['stopped_stage'] not in stages_not_saving:
        if param_dict['write_bathymetry']:
            write_bathymetry_data(out_nc, output_dict, fill_value, nb_pts_bathy_max=nb_pts_bathy_max)
        if param_dict['write_densification']:
            write_densification_groups(out_nc, output_dict, fill_value)
    else:
        logging.info(f'Skipping saving of bathymetry + densification for stopped stage: {output_dict['stopped_stage']}')
    if param_dict['gnuplot_saving']:
        reach_id = str(reach_id)
        output_dir = output_path.parent.joinpath('gnuplot_data', reach_id)
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True, exist_ok=True)
        gnuplot_save_q(q_algo31[:], times_var[:], output_dir.joinpath('qalgo31'))
    out_nc.close()

def write_algo5_params(out_nc, nc_dict, algo5_results, reach_number, output_dict, param_dict, fill_value):
    """Write Algo5 parameters (A0, n) to NetCDF file."""
    nc_dict = {}
    if algo5_results:
        for key in algo5_results.keys():
            key_name = 'A0' if key.lower() == 'a0' else 'n' if key.lower() == 'n' else key
            nc_dict[key] = out_nc.createVariable(key_name, 'f8', fill_value=fill_value)
    else:
        nc_dict['A0'] = out_nc.createVariable('A0', 'f8', fill_value=fill_value)
        nc_dict['n'] = out_nc.createVariable('n', 'f8', fill_value=fill_value)
    if param_dict['run_type'] == 'seq':
        if algo5_results:
            for key in nc_dict.keys():
                nc_dict[key].assignValue(algo5_results.get(key, np.nan))
        else:
            nc_dict['A0'].assignValue(np.nan)
            nc_dict['n'].assignValue(np.nan)
    elif reach_number < 0:
        nc_dict['A0'].assignValue(np.nan)
        nc_dict['n'].assignValue(np.nan)
    else:
        node_a0 = output_dict.get('node_a0', [])
        node_n = output_dict.get('node_n', [])
        nc_dict['A0'].assignValue(node_a0[reach_number] if node_a0 and reach_number < len(node_a0) else np.nan)
        nc_dict['n'].assignValue(node_n[reach_number] if node_n and reach_number < len(node_n) else np.nan)
    if params.tmp_write_to_A0_and_n:
        if 'Kmi_acc' in output_dict.keys() and 'Zb_acc' in output_dict.keys():
            if check_na(output_dict['Kmi_acc']) and check_na(output_dict['Zb_acc']):
                nc_dict['n'].assignValue(output_dict['Kmi_acc'])
                nc_dict['A0'].assignValue(output_dict['Zb_acc'])
                logging.warning('Kmi_acc or Zb_acc contains NaN')
            else:
                kmi = output_dict['Kmi_acc']
                zb = output_dict['Zb_acc']
                nc_dict['n'].assignValue(kmi[-1] if isinstance(kmi, (list, tuple)) else kmi)
                nc_dict['A0'].assignValue(zb[-1] if isinstance(zb, (list, tuple)) else zb)
                logging.info('Temporarily writing Kmi_acc and Zb_acc to A0 and n.')
        else:
            logging.warning('Kmi_acc and Zb_acc not found in output_dict for temporary writing to A0 and n.')
    for key in nc_dict.keys():
        logging.info(f'{key}: {nc_dict[key][:]}')

def write_discharge(out_nc, output_dict, reach_number, param_dict, fill_value):
    """Write discharge variables Q_mm (algo5) and Q_da (algo31)."""
    n_time = output_dict['time'].shape[0]
    q_algo5 = out_nc.createVariable('Q_mm', 'f8', ('nt',), fill_value=fill_value)
    if param_dict['run_type'] == 'seq':
        q_algo5[:] = output_dict.get('q_algo5', np.full(n_time, np.nan))
    elif reach_number < 0:
        q_algo5[:] = np.full(n_time, np.nan)
    else:
        q_algo5_all = output_dict.get('q_algo5_all', [])
        if q_algo5_all and reach_number < len(q_algo5_all):
            q_algo5[:] = q_algo5_all[reach_number]
        else:
            q_algo5[:] = np.full(n_time, np.nan)
    q_algo31 = out_nc.createVariable('Q_da', 'f8', ('nt',), fill_value=fill_value)
    q_algo31[:] = output_dict.get('q_algo31', np.full(n_time, np.nan))
    q_u_var = out_nc.createVariable('q_u', 'f8', ('nt',), fill_value=fill_value)
    q_u_var[:] = np.full(n_time, fill_value)
    logging.info(f'Q_da: {np.ma.count(q_algo31[:])}/{q_algo31[:].size} valid values')
    logging.info(f'Q_mm: {np.ma.count(q_algo5[:])}/{q_algo5[:].size} valid values')
    return (q_algo31, q_algo5)

def write_bathymetry_data(out_nc, output_dict, fill_value, nb_pts_bathy_max):
    """Write bathymetry variables (width, elevation, q_u)."""
    if 'width' in output_dict and 'elevation' in output_dict:
        output_dict['width'] = change_info_for_bathy(output_dict['width'], nb_pts_bathy_max, output_dict.get('node_id', []), output_dict.get('observed_nodes', []))
        output_dict['elevation'] = change_info_for_bathy(output_dict['elevation'], nb_pts_bathy_max, output_dict.get('node_id', []), output_dict.get('observed_nodes', []))
        width_data = np.array([np.pad(arr, (0, nb_pts_bathy_max - len(arr)), constant_values=np.nan) for arr in output_dict['width']])
        elevation_data = np.array([np.pad(arr, (0, nb_pts_bathy_max - len(arr)), constant_values=np.nan) for arr in output_dict['elevation']])
        if 'apr_array' in output_dict:
            observed_nodes = output_dict.get('observed_nodes', [])
            if np.array(observed_nodes).size == 0:
                logging.warning('No observed nodes found in output_dict for apr_array processing.')
            else:
                dry_area_data = change_info_for_bathy(output_dict['apr_array']['node_a'], len(output_dict['apr_array']['node_a'][0]), output_dict.get('node_id', []), output_dict.get('observed_nodes', []))
                hydraulic_radius_data = change_info_for_bathy(output_dict['apr_array']['node_r'], len(output_dict['apr_array']['node_r'][0]), output_dict.get('node_id', []), output_dict.get('observed_nodes', []))
        else:
            dry_area_data = np.full((len(output_dict.get('node_id', np.array([]))), len(output_dict['q_algo31'])), np.nan)
            hydraulic_radius_data = np.full((len(output_dict.get('node_id', np.array([]))), len(output_dict['q_algo31'])), np.nan)
    else:
        n_nodes = len(output_dict.get('node_id', np.array([])))
        width_data = np.full((n_nodes, nb_pts_bathy_max), np.nan)
        elevation_data = np.full((n_nodes, nb_pts_bathy_max), np.nan)
        dry_area_data = np.full((n_nodes, len(output_dict['q_algo31'])), np.nan)
        hydraulic_radius_data = np.full((n_nodes, len(output_dict['q_algo31'])), np.nan)
    width_var = out_nc.createVariable('width', 'f8', ('nodes', 'nb_pts'), fill_value=fill_value)
    elevation_var = out_nc.createVariable('elevation', 'f8', ('nodes', 'nb_pts'), fill_value=fill_value)
    dry_area_var = out_nc.createVariable('dry_area', 'f8', ('nodes', 'nt'), fill_value=fill_value)
    hydraulic_radius_var = out_nc.createVariable('hydraulic_radius', 'f8', ('nodes', 'nt'), fill_value=fill_value)
    width_var[:] = width_data
    elevation_var[:] = elevation_data
    dry_area_var[:] = dry_area_data
    hydraulic_radius_var[:] = hydraulic_radius_data
    logging.debug(f'bathymetry width: {np.ma.count(width_var[:])}/{width_var[:].size}')
    logging.debug(f'bathymetry elevation: {np.ma.count(elevation_var[:])}/{elevation_var[:].size}')
    logging.debug(f'bathymetry dry area: {np.ma.count(dry_area_var[:])}/{dry_area_var[:].size}')
    logging.debug(f'bathymetry hydraulic radius: {np.ma.count(hydraulic_radius_var[:])}/{hydraulic_radius_var[:].size}')
    reach_width, reach_elev, _ = aggregate_node_bathy_to_reach(elevation_data, width_data, option='all', nb_points=10)
    width_var = out_nc.createVariable('reach_width', 'f8', 'nb_pts', fill_value=fill_value)
    elevation_var = out_nc.createVariable('reach_elevation', 'f8', 'nb_pts', fill_value=fill_value)
    width_var[:] = reach_width
    elevation_var[:] = reach_elev
    if 'z_bed' in output_dict:
        z_bed = change_info_for_bathy(output_dict['z_bed'], 1, output_dict.get('node_id', []), output_dict.get('observed_nodes', []))
        z_bed_nc = out_nc.createVariable('z_bed', 'f8', 'nodes', fill_value=fill_value)
        z_bed_nc[:] = z_bed
    else:
        z_bed = np.full(len(output_dict['node_id']), np.nan)
        z_bed_nc = out_nc.createVariable('z_bed', 'f8', 'nodes', fill_value=fill_value)
        z_bed_nc[:] = z_bed
    A0 = compute_wet_area(width_data, elevation_data, z_bed)
    wet_area_nc = out_nc.createVariable('wet_area', 'f8', 'nodes', fill_value=fill_value)
    wet_area_nc[:] = A0
    width_min = np.nanmin(width_data, axis=1)
    width_min_nc = out_nc.createVariable('width_min', 'f8', 'nodes', fill_value=fill_value)
    width_min_nc[:] = width_min

def write_densification_groups(out_nc, output_dict, fill_value):
    """Write node and reach groups to NetCDF file."""
    group_node = out_nc.createGroup('node')
    node_id_var = group_node.createVariable('node_id', 'i8', ('nx',))
    node_ids_u2d = output_dict.get('node_id', [])
    node_ids_d2u = node_ids_u2d[::-1]
    node_id_var[:] = node_ids_d2u
    reach_t_dup = output_dict.get('reach_t_duplicate', output_dict.get('reach_t', np.array([])))
    time_var = group_node.createVariable('time', 'f8', ('nx', 'nt'), fill_value=-999999999999.0)
    time_var[:, :] = reach_t_dup if reach_t_dup.size > 0 else np.full((output_dict.get('node_z', np.array([])).shape[0], output_dict['time'].shape[0]), np.nan)
    wse_var = group_node.createVariable('wse', 'f8', ('nx', 'nt'), fill_value=-999999999999.0)
    wse_var.units = 'm'
    node_wse_u2d = output_dict.get('node_z', np.full((output_dict.get('nx', 1), output_dict['time'].shape[0]), np.nan))
    node_wse_d2u = node_wse_u2d[::-1] if node_wse_u2d.size > 0 else np.full((output_dict.get('nx', 1), output_dict['time'].shape[0]), np.nan)
    wse_var[:, :] = node_wse_d2u
    width_var = group_node.createVariable('width', 'f8', ('nx', 'nt'), fill_value=-999999999999.0)
    width_var.units = 'm'
    node_width_u2d = output_dict.get('node_w', np.full((output_dict.get('nx', 1), output_dict['time'].shape[0]), np.nan))
    node_width_d2u = node_width_u2d[::-1] if node_width_u2d.size > 0 else np.full((output_dict.get('nx', 1), output_dict['time'].shape[0]), np.nan)
    width_var[:, :] = node_width_d2u
    slope_var = group_node.createVariable('slope', 'f8', ('nx', 'nt'), fill_value=-999999999999.0)
    slope_var.units = 'm/m'
    slope_var[:, :] = np.nan
    group_node.setncattr('description', 'Densification results at node level')
    logging.debug(f'densification node wse: {np.ma.count(wse_var[:])}/{wse_var[:].size}')
    logging.debug(f'densification node width: {np.ma.count(width_var[:])}/{width_var[:].size}')
    if 'reach_slope' in output_dict.keys():
        reach_wse, reach_width, time_in_seconds, reach_time_str, reach_slope = aggregate_to_reach_level(output_dict['node_z'], output_dict['node_w'], output_dict['time'], reach_slope=output_dict['reach_slope'])
    else:
        reach_wse, reach_width, time_in_seconds, reach_time_str, reach_slope = aggregate_to_reach_level(output_dict['node_z'], output_dict['node_w'], output_dict['time'])
    group_reach = out_nc.createGroup('reach')
    reach_time_var = group_reach.createVariable('time', 'f8', ('nt',), fill_value=-999999999999.0)
    reach_time_var.long_name = 'time (UTC)'
    reach_time_var.units = 'seconds since 2000-01-01 00:00:00.000'
    reach_time_var.calendar = 'gregorian'
    reach_time_var[:] = time_in_seconds
    group_reach.createDimension('chartime', 20)
    reach_time_str_var = group_reach.createVariable('time_str', 'S1', ('nt', 'chartime'), fill_value=b'n')
    reach_time_str_var.long_name = 'UTC time'
    reach_time_str_var.standard_name = 'time'
    reach_time_str_var.calendar = 'gregorian'
    reach_time_str_var.comment = 'Time string giving UTC time. The format is YYYY-MM-DDThh:mm:ssZ.'
    reach_time_str_var[:] = reach_time_str
    reach_wse_var = group_reach.createVariable('wse', 'f8', ('nt',), fill_value=-999999999999.0)
    reach_wse_var.units = 'm'
    reach_wse_var[:] = reach_wse
    reach_width_var = group_reach.createVariable('width', 'f8', ('nt',), fill_value=-999999999999.0)
    reach_width_var.units = 'm'
    reach_width_var[:] = reach_width
    reach_slope_var = group_reach.createVariable('slope', 'f8', ('nt',), fill_value=-999999999999.0)
    reach_slope_var.units = 'm/m'
    reach_slope_var[:] = reach_slope
    reach_slope_var = group_reach.createVariable('slope2', 'f8', ('nt',), fill_value=-999999999999.0)
    reach_slope_var.units = 'm/m'
    reach_slope_var[:] = reach_slope
    group_reach.setncattr('description', 'Densification results at reach level')
    logging.info(f'densification reach wse: {np.ma.count(reach_wse_var[:])}/{reach_wse_var[:].size}')
    logging.info(f'densification reach width: {np.ma.count(reach_width_var[:])}/{reach_width_var[:].size}')

def read_bathymetry(bathymetry_path, dist):
    width = []
    wse = []
    temp1 = []
    temp2 = []
    indexes = []
    with open(bathymetry_path, 'r') as file:
        j = 0
        for line in file:
            words = line.split()
            '\n            # Check the length of the words list to determine the structure\n            if len(words) == 11:\n                # This structure has 11 elements\n                # Parse the elements as needed\n                element1 = words[0]\n                element2 = float(words[1])  # Convert to float if needed\n                # ... continue parsing the remaining elements\n            '
            if len(words) == 2:
                element1 = float(words[0])
                element2 = float(words[1])
                temp1.append(element1)
                temp2.append(element2)
            if len(words) != 2:
                if len(words) >= 11:
                    index = []
                    '\n                    for l in range(0,len(dist)):\n                        if np.array(index).size ==0:\n                            index = np.where( np.isclose( float(words[2]), float(dist[l]) ) )\n                        elif np.array(index).size > 0:\n                            break \n                    print(index)\n                    '
                    index = np.where(np.isclose(float(words[2]), dist))
                    if np.array(temp1).size > 0 and np.array(temp2).size > 0:
                        width.append(temp1)
                        wse.append(temp2)
                        temp1 = []
                        temp2 = []
                        j = len(wse)
                        if np.array(index[0]).size > 0:
                            indexes.append(j)
                    '\n                    if is_integer(words[0]):\n                        #if int(words[0])==12:\n                    '
                    if np.array(index[0]).size > 0:
                        pass
    return (width, wse, indexes)

def sword_table(_input_dir, _reach_dict):
    sword_file = _input_dir.joinpath('sword', f'{_reach_dict['sword']}')
    sword_file_exists = os.path.isfile(sword_file)
    if sword_file_exists:
        sword_dataset = Dataset(sword_file)
        id_ctl = sword_dataset['centerlines/cl_id'][:]
        nb_nod = sword_dataset['nodes/node_id'][:].size
        nb_rch = sword_dataset['reaches/reach_id'][:].size
        id_nod_ctl = sword_dataset['nodes/cl_ids'][:]
        id_rch_ctl = sword_dataset['reaches/cl_ids'][:]
        sword_dataset.close()
    min_ctr_id = np.min(id_ctl)
    max_ctr_id = np.max(id_ctl)
    nbr_ctr_id = max_ctr_id - min_ctr_id + 1
    index_ctl_ctl = np.ones(nbr_ctr_id) * -1
    nb_ctl = id_ctl.size
    for i in range(0, nb_ctl):
        index_ctl_ctl[id_ctl[i] - min_ctr_id] = i
    index_nod_ctl = np.ones(nbr_ctr_id) * -1
    for i in range(0, nb_nod):
        for j in range(id_nod_ctl[0][i], id_nod_ctl[1][i] + 1):
            index_nod_ctl[j - min_ctr_id] = i
    index_rch_ctl = np.ones(nbr_ctr_id) * -1
    for i in range(0, nb_rch):
        for j in range(id_rch_ctl[0][i], id_rch_ctl[1][i] + 1):
            index_rch_ctl[j - min_ctr_id] = i
    index_rch_node = np.zeros((nb_rch, 2))
    for i in range(0, nb_rch):
        for j in range(0, 2):
            if i == 15348:
                pass
            index_rch_node[i, j] = index_nod_ctl[id_rch_ctl[j, i] - min_ctr_id]
    np.save('index_rch_node', index_rch_node)

def get_run_type(json_path):
    if not json_path.exists() or not json_path.suffix == '.json':
        raise IOError('Unable to load json file %s' % json_path)
    with open(json_path) as jsonfile:
        json_data = json.load(jsonfile)
    run_type = 'seq'
    for elem in json_data:
        if isinstance(elem, list):
            run_type = 'set'
    return run_type

def usgs_q(id_test, ds, ds2, timestamp1, timestamp2):

    def usgs_nonNaN(id_test, ds, timestamp1, timestamp2):

        def usgs_id_decode(data, index):
            var1 = data['model/usgs/usgs_id'][index].astype('U13')[0]
            var1 = int(''.join(var1))
            return var1

        def usgs_id_encode(integer):
            arr = np.pad([*str(testint)], (0, 8), 'constant', constant_values='')
            arr = np.char.encode(arr, encoding='UTF-8')
            return arr
        timetmp1 = timestamp1
        timetmp2 = timestamp2
        timestamp1 = int(datetime2matlabdn(timestamp1))
        timestamp2 = int(datetime2matlabdn(timestamp2))
        jj_usgs = np.where(ds['model']['usgs']['usgs_reach_id'][:] == id_test)
        len_jj_usgs = np.array(jj_usgs).size
        jj_usgs = np.array(jj_usgs)[0]
        q_threshold_ohio = 5
        mean_q_usgs_sos = ds['model']['usgs']['mean_q'][:]
        nb_station_usgs_case = 0
        jj_usgs_qNaN = []
        jj_usgs_qtNaN = []
        if len_jj_usgs >= 1:
            times = ds['model']['usgs']['usgs_qt'][jj_usgs]
            good_usgs_station = []
            nb_station_usgs_case = 0
            for jj in range(0, len_jj_usgs):
                good_usgs_station.append(0)
                index_time = np.where((ds['model']['usgs']['usgs_qt'][jj_usgs[jj]] >= timestamp1) & (ds['model']['usgs']['usgs_qt'][jj_usgs[jj]] <= timestamp2))
                jj_usgs_qNaN = ds['model']['usgs']['usgs_q'][jj_usgs[jj]]
                index_nonNaN = np.where(jj_usgs_qNaN.data > 0)
                all_nonNaN = np.intersect1d(index_time, index_nonNaN)
                jj_usgs_qNaN = jj_usgs_qNaN[all_nonNaN]
                jj_usgs_qtNaN = ds['model']['usgs']['usgs_qt'][jj_usgs[jj]]
                jj_usgs_qtNaN = jj_usgs_qtNaN[all_nonNaN]
                if len(jj_usgs_qNaN) > 0:
                    if np.any(np.array(np.where(jj_usgs_qNaN > 0))[0]):
                        good_usgs_station[jj] += 1
                    if mean_q_usgs_sos[jj_usgs[jj]] > q_threshold_ohio:
                        good_usgs_station[jj] += 1
                else:
                    pass
                val_test = max(good_usgs_station)
                jj_usgs_best = jj_usgs[jj]
                if val_test == 2:
                    nb_station_usgs_case += 1
                    ind_usgs = jj_usgs_best
                else:
                    ind_usgs = 0
        else:
            ind_usgs = 0
        if nb_station_usgs_case > 0:
            pass
        else:
            pass
        return (jj_usgs_qNaN, jj_usgs_qtNaN)

    def q_array(ds2, out):
        timestamp2000 = 946684800
        q_array = []
        for i in range(0, np.array(ds2['reach']['time'][:]).size):
            test_date = int(ds2['reach']['time'][i]) + timestamp2000
            str_test = datetime.datetime.fromtimestamp(test_date).strftime('%A %B %d %Y %I:%M:%S')
            testo = datetime.datetime.strptime(str_test, '%A %B %d %Y %I:%M:%S')
            timestamp3 = int(datetime2matlabdn(testo))
            index_times2 = np.where(out[1][:] == int(timestamp3))
            q_array.append(out[0][index_times2])
        q_array = np.array(q_array)
        q_array = q_array.reshape(1, q_array.size)[0]
        return q_array
    out = usgs_nonNaN(id_test, ds, timestamp1, timestamp2)
    if np.array(out).size != 0:
        q_array = q_array(ds2, out)
        return q_array
    return []

def width_replacement(node_w, reach_w, node_w_mean, reach_w_mean):
    option = 1
    condition_reach = []
    if len(node_w[0]) != len(reach_w) or len(node_w[0]) != len(node_w_mean):
        return (True, node_w)
    if option == 1:
        for t in range(0, len(reach_w)):
            condition_reach.append(calc.check_na(reach_w[t]))
        reach_index_nobs = np.where((reach_w == 0) | condition_reach)
        if np.array(reach_index_nobs).size == len(reach_w):
            return (False, [])
        for t in range(0, len(reach_w)):
            condition2 = []
            for n in range(0, len(node_w)):
                condition2.append(calc.check_na(node_w[n, t]))
            condition = (node_w[:, t] == 0) | condition2
            index_to_replace = np.array(np.where(condition))
            index_to_replace1 = np.array(np.where(node_w[:, t] == 0))
            index_to_replace2 = np.array(np.where(condition2))
            if np.array(index_to_replace).size > 0:
                if not calc.check_na(reach_w[t]) or reach_w[t] > 0.0:
                    node_w[index_to_replace, t] = reach_w[t] * (node_w_mean[t] / reach_w_mean)
                else:
                    pass
    return (True, node_w)

def correlation_nodes(node_z, ref_node, reach_id):
    indicators_nodes = {}
    ref_array = node_z[ref_node, :]
    ref_idx = []
    for t in range(0, len(node_z[0])):
        if calc.check_na(node_z[0, t]):
            pass
        else:
            ref_idx.append(t)
    print('LEN REF ARRAY T:', len(ref_array))
    from lib.lib_indicators import compute_all_indicators_from_predict_true
    for n in range(0, len(node_z)):
        idx_non_nan = []
        for t in range(0, len(node_z[n])):
            if calc.check_na(node_z[n, t]):
                pass
            else:
                idx_non_nan.append(t)
        common_indexes = np.intersect1d(ref_idx, idx_non_nan)
        print(f'common_indexes in t for node {n}:', len(common_indexes))
        indicators_nodes[n] = {'reach_id': str(reach_id), 'nb_dates': len(common_indexes)}
        indicators_nodes[n].update(compute_all_indicators_from_predict_true(ref_array[common_indexes], node_z[n, common_indexes]))
    indicators_node_df = pd.DataFrame(indicators_nodes).transpose()
    indicators_node_df.to_csv(Path('/home/kabey/Bureau/worksync/Pankaj_Ganga/output/nodes_corr_z' + '.csv'), index=True, sep=';')
    print(bug2)

def use_stations_for_q_prior(pred_times, source_discharge, source_times, source_times_in_days):
    df = pd.DataFrame({'source_times': source_times, 'source_q': source_discharge, 'source_times_in_days': source_times_in_days})
    pred_times_array = np.ma.asarray(pred_times)
    if np.ma.isMaskedArray(pred_times_array):
        pred_times_values = pred_times_array.compressed()
    else:
        pred_times_values = np.asarray(pred_times_array).ravel()
    pred_times_series = pd.Series(pred_times_values, name='pred_times')
    pred_times_series = pd.to_datetime(pred_times_series, errors='coerce').dropna()
    if pred_times_series.size < 1:
        logging.warning(call_error_message(503))
        return (-9999.0, -9999.0, pd.DataFrame())
    result = []
    tmp_option = 1
    if tmp_option == 0:
        ref = np.datetime64('2000-01-01')
        seconds_since_2000 = (pred_times_series - ref) / np.timedelta64(1, 's')
        swot_days = np.array(seconds_since_2000 / 86400).astype(int)
        swot_df = pd.DataFrame({'pred_times': pred_times_series, 'source_times_in_days': swot_days})
        filtered_df = pd.merge(df, swot_df, on='source_times_in_days', how='inner')
    elif tmp_option == 1:
        if not params.force_specific_dates:
            start_date = pred_times_series.min()
            end_date = pred_times_series.max()
        else:
            start_date = params.start_date.date()
            end_date = params.end_date.date()
        filtered_df = df[(df['source_times'] >= start_date) & (df['source_times'] <= end_date)]
        filtered_df = filtered_df.dropna(subset=['source_q'], how='all')
        if len(filtered_df['source_q']) < 2:
            logging.warning(call_error_message(502))
            return (-9999.0, -9999.0, pd.DataFrame())
        if False:
            if params.smoothing_prior:
                cor_radius_t = (sic_df['sic4dvar_t'].iloc[-1] - sic_df['sic4dvar_t'].iloc[0]) / (len(sic_df) - 1) / 86400
                discharge_smoothed_2D = []
                for n in range(0, 2):
                    discharge_smoothed_2D.append(filtered_station_df['station_q'])
                discharge_smoothed_2D = np.array(discharge_smoothed_2D)
                discharge_smoothed_2D = K(dim=0, value0_array=discharge_smoothed_2D, base0_array=np.array(filtered_station_df[string]), max_iter=1, cor=cor_radius_t, always_run_first_iter=True, behavior='', inter_behavior=False, inter_behavior_min_thr=params.def_float_atol, inter_behavior_max_thr=params.DX_max_in, check_behavior='', min_change_v_thr=0.0001, plot=False, plot_title='', clean_run=True, debug_mode=False, time_integration=False)
                discharge_smoothed = discharge_smoothed_2D[0]
                filtered_station_df['station_q'] = discharge_smoothed
    from lib.lib_indicators import integrated_mean, integrated_variance
    source_q_arr = filtered_df['source_q'].to_numpy()
    source_t_arr = filtered_df['source_times_in_days'].to_numpy()
    q_mean = integrated_mean(source_q_arr, source_t_arr)
    q_std = integrated_variance(source_q_arr, source_t_arr, q_mean)
    logging.debug('Computed station prior stats: q_mean=%s q_std=%s', q_mean, q_std)
    return (q_mean, q_std, filtered_df)

def use_stations_for_q_prior_old(reach_t, station_q, station_qt, reach_id=[], station_qt2=[], option='station', param_dict=[], option2=1, additional_t=[]):
    station_df = pd.DataFrame({'station_date': station_qt, 'station_q': station_q, 'station_qt': station_qt2})
    if option == 'station':
        epoch_0001 = datetime(1, 1, 1)
        epoch_2000 = datetime(2000, 1, 1)
        delta_days = (epoch_2000 - epoch_0001).days
        string = 'station_qt_2000'
        station_df['station_qt_2000_days'] = [v - delta_days for v in station_df['station_qt']]
        station_df[string] = station_df['station_qt_2000_days'] * 86400
    elif option == 'ML':
        string = 'ML_qt_2000'
        station_df[string] = station_df['station_qt']
    if reach_t.mask.any():
        sic4dvar_valid_idx = np.where(reach_t.mask == False)
        sic4dvar_t = reach_t[sic4dvar_valid_idx]
    else:
        sic4dvar_t = reach_t[:]
    if np.array(sic4dvar_t).size < 1:
        logging.warning(call_error_message(503))
        return (-9999.0, -9999.0, pd.DataFrame())
    dates_sic = []
    for date in sic4dvar_t:
        dates_sic.append(seconds_to_date_old(date))
    sic_df = pd.DataFrame({'sic4dvar_date': dates_sic, 'sic4dvar_t': sic4dvar_t})
    sic_df['date_only'] = pd.to_datetime(sic_df['sic4dvar_date'], format='%Y-%m-%d').dt.date
    station_df['date_only'] = pd.to_datetime(station_df['station_date'], format='%Y-%m-%d').dt.date
    result = []
    if option2 == 0:
        filtered_station_df = pd.merge(station_df, sic_df, on='date_only', how='inner')
        if len(filtered_station_df['station_q']) < 2:
            return (-9999.0, -9999.0, pd.DataFrame())
    if option2 == 1:
        if not params.force_specific_dates:
            start_date = sic_df['date_only'].min()
            end_date = sic_df['date_only'].max()
        else:
            start_date = params.start_date.date()
            end_date = params.end_date.date()
        filtered_station_df = station_df[(station_df['date_only'] >= start_date) & (station_df['date_only'] <= end_date)]
        if params.use_SVS:
            filtered_station_df = filtered_station_df.dropna(subset=['station_q'], how='all')
        if len(filtered_station_df['station_q']) < 2:
            logging.warning(call_error_message(502))
            return (-9999.0, -9999.0, pd.DataFrame())
    elif option2 == 2:
        station_df_orig = deepcopy(station_df)
        if additional_t.mask.any():
            sic4dvar_valid_idx = np.where(additional_t.mask == False)
            sic4dvar_t = additional_t[sic4dvar_valid_idx]
        else:
            sic4dvar_t = additional_t[:]
        if np.array(sic4dvar_t).size < 1:
            logging.warning(call_error_message(503))
            return (-9999.0, -9999.0, pd.DataFrame())
        times_sic = np.array(sic4dvar_t / 86400)
        time_system = times_sic
        if len(time_system) < 2:
            logging.info("Time system size < 2, can't compute integrated estimated mean.")
            return (-9999.0, -9999.0, pd.DataFrame())
        q_ref_ML = []
        t_ref = []
        for t in range(0, len(time_system)):
            q_ref_ML.append(interp_pdf_tables(len(station_df['station_q']) - 1, time_system[t], np.array(station_df[string]), np.array(station_df['station_q'])))
            t_ref.append(time_system[t])
        station_df = pd.DataFrame({'ML_qt_2000': t_ref, 'station_q': q_ref_ML, 'station_qt_2000': t_ref, 'station_qt': t_ref})
        dates_sic = []
        for date in station_df['ML_qt_2000'] * 86400:
            dates_sic.append(seconds_to_date_old(date))
        station_df['dates'] = dates_sic
        station_df['date_only'] = pd.to_datetime(station_df['dates'], format='%Y-%m-%d').dt.date
        if not params.force_specific_dates:
            start_date = station_df['date_only'].min()
            end_date = station_df['date_only'].max()
        else:
            start_date = params.start_date.date()
            end_date = params.end_date.date()
        start_date_ML = station_df_orig['date_only'].min()
        end_date_ML = station_df_orig['date_only'].max()
        if start_date > start_date_ML:
            pass
        if end_date > end_date_ML:
            end_date = end_date_ML
        filtered_station_df = station_df[(station_df['date_only'] >= start_date) & (station_df['date_only'] <= end_date)]
        if len(filtered_station_df['station_q']) < 2:
            return (-9999.0, -9999.0, pd.DataFrame())
    q_mean = 0
    time_scaling = 0
    q_std = 0.0
    if option == 'ML':
        if params.smoothing_prior:
            cor_radius_t = (sic_df['sic4dvar_t'].iloc[-1] - sic_df['sic4dvar_t'].iloc[0]) / (len(sic_df) - 1) / 86400
            discharge_smoothed_2D = []
            for n in range(0, 2):
                discharge_smoothed_2D.append(filtered_station_df['station_q'])
            discharge_smoothed_2D = np.array(discharge_smoothed_2D)
            discharge_smoothed_2D = K(dim=0, value0_array=discharge_smoothed_2D, base0_array=np.array(filtered_station_df[string]), max_iter=1, cor=cor_radius_t, always_run_first_iter=True, behavior='', inter_behavior=False, inter_behavior_min_thr=params.def_float_atol, inter_behavior_max_thr=params.DX_max_in, check_behavior='', min_change_v_thr=0.0001, plot=False, plot_title='', clean_run=True, debug_mode=False, time_integration=False)
            discharge_smoothed = discharge_smoothed_2D[0]
            filtered_station_df['station_q'] = discharge_smoothed
    for t in range(1, len(filtered_station_df[string])):
        q_mean += (filtered_station_df['station_q'].iloc[t] + filtered_station_df['station_q'].iloc[t - 1]) / 2 * (filtered_station_df[string].iloc[t] - filtered_station_df[string].iloc[t - 1])
        time_scaling += np.array(filtered_station_df[string].iloc[t] - filtered_station_df[string].iloc[t - 1])
    if time_scaling == 0:
        raise ValueError(f'time_scaling is zero for REACH_ID {reach_id}')
    q_mean = q_mean / time_scaling
    for t in range(1, len(filtered_station_df[string])):
        q_std += ((filtered_station_df['station_q'].iloc[t] - q_mean) ** 2 + (filtered_station_df['station_q'].iloc[t - 1] - q_mean) ** 2) / 2 * (filtered_station_df[string].iloc[t] - filtered_station_df[string].iloc[t - 1])
    q_std = (q_std / time_scaling) ** (1 / 2)
    return (q_mean, q_std, filtered_station_df)

def define_spread(q_mean, q_std, use_mean_for_bounds, Qp=[], quantiles=[], relative_variance=2.0):
    epsilon = 0.1
    if use_mean_for_bounds:
        QM1 = q_mean / Qp
        QM2 = q_mean * Qp
    else:
        QM1 = quantiles[-1]
        QM2 = quantiles[0]
    mu = (q_mean - QM1) / (QM2 - QM1)
    beta = mu * (1 - mu) / ((q_std / (QM2 - QM1)) ** 2 * (1 / (1 + relative_variance))) - 1
    if beta < 1 / np.nanmin([mu, 1 - mu]) + epsilon:
        beta = 1 / np.nanmin([mu, 1 - mu]) + epsilon
    elif beta > 50:
        beta = 50
    return (beta, QM1, QM2)

def gamma_table(beta):
    beta_array = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    gamma_array = [1.497, 1.371, 1.295, 1.244, 1.208, 1.181, 1.16, 1.144, 1.13, 1.119, 1.11, 1.102, 1.095, 1.089, 1.084, 1.079, 1.075, 1.07, 1.057, 1.047, 1.041, 1.035, 1.031, 1.028, 1.025, 1.023]
    gamma = interp_pdf_tables(len(beta_array) - 1, beta, beta_array, gamma_array)
    return gamma

def build_output_q_masked_array(sic4dvar_dict, str):
    """Build output array, including mask and replacing the NaNs that were removed.

    Parameters
    ----------
    str : string
        The key of the ouput dictionary, either "q_algo31" or "q_algo5"

    Returns
    -------
    array
        returns array of prepared masked array.
    """
    if str == 'q_algo31' or str == 'time':
        mask = np.ones(sic4dvar_dict['input_data']['node_z'].shape[1], dtype=bool)
        arr = np.ones(sic4dvar_dict['input_data']['node_z'].shape[1]) * np.nan
        if np.array(sic4dvar_dict['list_to_keep']).size > 0:
            logging.debug(f'list_to_keep: {sic4dvar_dict['list_to_keep']}')
            if len(sic4dvar_dict['list_to_keep']) == len(sic4dvar_dict['output'][str]):
                arr[sic4dvar_dict['list_to_keep']] = sic4dvar_dict['output'][str]
            else:
                arr[sic4dvar_dict['list_to_keep']] = sic4dvar_dict['output'][str][sic4dvar_dict['list_to_keep']]
        if np.array(sic4dvar_dict['removed_indices']).size > 0:
            arr[sic4dvar_dict['removed_indices']] = np.nan
    elif str == 'q_algo5':
        mask = np.ones(sic4dvar_dict['input_data']['node_z'].shape[1], dtype=bool)
        arr = np.ones(sic4dvar_dict['input_data']['node_z'].shape[1]) * np.nan
        if np.array(sic4dvar_dict['list_to_keep']).size > 0:
            if len(sic4dvar_dict['list_to_keep']) == len(sic4dvar_dict['output'][str]):
                arr[sic4dvar_dict['list_to_keep']] = sic4dvar_dict['output'][str]
            else:
                arr[sic4dvar_dict['list_to_keep']] = sic4dvar_dict['output'][str][sic4dvar_dict['list_to_keep']]
        index_tmp = np.where(sic4dvar_dict['output'][str] <= 0.0)
        arr[index_tmp[0]] = np.nan
    for ind in range(0, len(arr)):
        if not calc.check_na(arr[ind]):
            mask[ind] = False
    q_masked = np.ma.array(arr, mask=mask)
    return q_masked

def build_output_2d_masked_array(sic4dvar_dict, arr_2d, set_non_positive_to_nan=False):
    """Build a 2D output array on the original time axis, with mask.

    Parameters
    ----------
    arr_2d : array-like
        Array with shape (n_nodes, n_kept_times) or (n_nodes, n_total_times).
    set_non_positive_to_nan : bool
        If True, values <= 0 are set to NaN before masking.

    Returns
    -------
    array
        Returns rebuilt masked array of shape (n_nodes, n_total_times).
    """
    arr_2d = np.array(arr_2d, dtype=float)
    n_total_times = sic4dvar_dict['input_data']['node_z'].shape[1]
    n_nodes = arr_2d.shape[0]
    arr = np.full((n_nodes, n_total_times), np.nan, dtype=float)
    if np.array(sic4dvar_dict['list_to_keep']).size > 0:
        if len(sic4dvar_dict['list_to_keep']) == arr_2d.shape[1]:
            arr[:, sic4dvar_dict['list_to_keep']] = arr_2d
        else:
            arr[:, sic4dvar_dict['list_to_keep']] = arr_2d[:, sic4dvar_dict['list_to_keep']]
    if np.array(sic4dvar_dict['removed_indices']).size > 0:
        arr[:, sic4dvar_dict['removed_indices']] = np.nan
    if set_non_positive_to_nan:
        arr[arr <= 0.0] = np.nan
    check_na_vector = np.vectorize(calc.check_na)
    mask = check_na_vector(arr)
    return np.ma.array(arr, mask=mask)

def aggregate_to_reach_level(node_wse, node_width, node_time, reach_length=None, reach_slope=None):
    reach_wse = np.nanmean(node_wse, axis=0) if np.array(node_wse).size > 0 else np.nan
    reach_width = np.nanmean(node_width, axis=0) if np.array(node_width).size > 0 else np.nan
    time_in_seconds = node_time * 86400.0
    reach_time_str = seconds_to_time_str(time_in_seconds)
    if reach_length:
        reach_slope = (node_wse[-1, :] - node_wse[0, :]) / reach_length
    return (reach_wse, reach_width, time_in_seconds, reach_time_str, reach_slope)

def make_mask_period(reach_time, list_months_to_use, plot=False, window_scale='months'):
    ref = np.datetime64('2000-01-01')
    swot_datetimes = ref + reach_time * np.timedelta64(1, 's')
    mask = np.zeros(reach_time.shape, dtype=bool)
    if window_scale == 'months':
        for t, swot_datetime in enumerate(swot_datetimes):
            month = swot_datetime.astype('datetime64[M]').astype(int) % 12 + 1
            if month in list_months_to_use:
                mask[t] = 1
            else:
                pass
        sigma = 0.01
        months = swot_datetimes.astype('datetime64[M]').astype(int) % 12 + 1
        weights = np.zeros_like(months, dtype=float)
        dist = np.abs(months - list_months_to_use[0])
        dist = np.minimum(dist, 12 - dist)
        weights = np.exp(-dist ** 2 / (2 * sigma ** 2))
    elif window_scale == 'days':
        year_begin = swot_datetimes.min().astype('datetime64[Y]').astype(int) + 1970
        year_end = swot_datetimes.max().astype('datetime64[Y]').astype(int) + 1970
        sigma = 16.0
        if False:
            swot_datetimes = []
            for j in range(1, 3):
                string_j = str(j)
                if j < 10:
                    string_j = f'0{j}'
                for i in range(1, 30):
                    string_i = str(i)
                    if i < 10:
                        string_i = f'0{i}'
                    if j == 2 and i > 27:
                        continue
                    swot_datetimes.append(f'2024-{string_j}-{string_i}T13:31:53')
            swot_datetimes = np.array(swot_datetimes, dtype='datetime64[s]')
        dist = np.zeros_like(swot_datetimes, dtype=float)
        for t in range(0, len(swot_datetimes)):
            year_swot_datetime = swot_datetimes[t].astype('datetime64[Y]').astype(int) + 1970
            if type(list_months_to_use[0]) == int and list_months_to_use[0] < 10:
                list_months_to_use[0] = f'0{list_months_to_use[0]}'
            ref_date = np.datetime64(f'{year_swot_datetime}-{list_months_to_use[0]}-15')
            dist[t] = np.abs(ref_date - swot_datetimes[t]) / np.timedelta64(1, 'D')
        weights = np.zeros_like(swot_datetimes, dtype=float)
        weights = np.exp(-(dist / sigma) ** 6)
    if plot:
        print(swot_datetimes)
        plt.plot(swot_datetimes, weights, 'bo')
        plt.show()
        plt.clf()
    return weights

def compute_mean_elevation_profile(node_z, reach_t):
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
    print('Mean profile array:', mean_elevation_profile_array)
    return mean_elevation_profile_array