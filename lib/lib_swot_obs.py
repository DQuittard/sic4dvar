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
import logging
import xml.etree.ElementTree as ET
import numpy as np
swot_quality_flags_list = ['xovr_cal_q', 'node_q', 'node_q_b', 'reach_q_b', 'dark_frac', 'obs_frac', 'partial_f', 'wse_r_u', 'ice_clim_f']
swot_xovr_cal_q_labels = {0: 'good', 1: 'suspect', 2: 'bad'}
swot_node_q_labels = {0: 'good', 1: 'suspect', 2: 'degraded', 3: 'bad'}
swot_node_q_labels = {0: 'good', 1: 'suspect', 2: 'degraded', 3: 'bad'}
swot_common_q_b_decimal_labels = {'classification_qual_suspect': 2, 'geolocation_qual_suspect': 4, 'water_fraction_suspect': 8, 'bright_land': 128, 'few_area_observations': 1024, 'few_wse_observations': 2048, 'far_range_suspect': 8192, 'near_range_suspect': 16384, 'classification_qual_degraded': 262144, 'geolocation_qual_degraded': 524288, 'lake_flagged': 4194304, 'no_area_observations': 67108864, 'no_wse_observations': 134217728, 'no_observations': 268435456}
swot_q_b_decimal_labels = {'node': {'sig0_qual_suspect': 1, 'blocking_width_suspect': 16, 'few_sig0_observations': 512, 'wse_outlier': 8388608, 'wse_bad': 16777216, 'no_sig0_observations': 33554432}, 'reach': {'partially observed': 32768, 'below min fit points': 33554432}}
swot_q_b_decimal_labels['node'].update(swot_common_q_b_decimal_labels)
swot_q_b_decimal_labels['reach'].update(swot_common_q_b_decimal_labels)
swot_common_q_b_bit_labels = {1: 'classification_qual_suspect', 2: 'geolocation_qual_suspect', 3: 'water_fraction_suspect', 7: 'bright_land', 10: 'few_area_observations', 11: 'few_wse_observations', 13: 'far_range_suspect', 14: 'near_range_suspect', 18: 'classification_qual_degraded', 19: 'geolocation_qual_degraded', 22: 'lake_flagged', 26: 'no_area_observations', 27: 'no_wse_observations', 28: 'no_observations'}
swot_q_b_bit_labels = {'node': {0: 'sig0_qual_suspect', 4: 'blocking_width_suspect', 9: 'few_sig0_observations', 23: 'wse_outlier', 24: 'wse_bad', 25: 'no_sig0_observations'}, 'reach': {15: 'partially observed', 25: 'below min fit points'}}
swot_q_b_bit_labels['node'].update(swot_common_q_b_bit_labels)
swot_q_b_bit_labels['reach'].update(swot_common_q_b_bit_labels)
swot_q_b_bit_labels['node'] = dict(sorted(swot_q_b_bit_labels['node'].items()))
swot_q_b_bit_labels['reach'] = dict(sorted(swot_q_b_bit_labels['reach'].items()))
swotfile_var_type = {'wse_r_u': 'float64', 'xtrk_dist': 'float64', 'slope_r_u': 'float64', 'slope2_r_u': 'float64', 'reach_q_b': 'float64'}

def remove_outliers_tukey(data, threshold=1.5):
    """
    Removes outliers from a dataset based on Tukey's IQR method.
    
    Parameters:
        data (array-like): Input data (1D list or NumPy array).
        threshold (float): Multiplication factor for the IQR (default is 1.5).
    
    Returns:
        np.ndarray: Filtered data without outliers.
    """
    data = np.array(data)
    Q1 = np.nanpercentile(data, 25)
    Q3 = np.nanpercentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    data[np.where(data < lower_bound)] = np.nan
    data[np.where(data > upper_bound)] = np.nan
    return data

def filter_swot_obs_on_quality(swot_dict_arrays, val_swot_q_flag):
    node_level_values_to_remove_mask = True * np.zeros((swot_dict_arrays['nx'], swot_dict_arrays['nt']))
    for q_var, q_val in val_swot_q_flag['node'].items():
        if q_var == 'xtrk_dist_max':
            var_mask = np.greater(np.abs(swot_dict_arrays['node']['xtrk_dist']), q_val).filled(np.nan)
        elif q_var == 'xtrk_dist_min':
            var_mask = np.less(np.abs(swot_dict_arrays['node']['xtrk_dist']), q_val).filled(np.nan)
        elif q_var == 'dark_frac_max':
            var_mask = np.greater(swot_dict_arrays['node']['dark_frac'], q_val).filled(np.nan)
        elif q_var == 'n_good_pix_min':
            var_mask = np.less(swot_dict_arrays['node']['n_good_pix'], q_val).filled(np.nan)
        elif q_var == 'wse_u_max':
            var_mask = np.greater(swot_dict_arrays['node']['wse_u'], q_val).filled(np.nan)
        elif q_var == 'width_u_max':
            var_mask = np.greater(swot_dict_arrays['node']['width_u'], q_val).filled(np.nan)
        elif q_var == 'node_q':
            var_mask = ~np.isin(swot_dict_arrays['node']['node_q'], q_val)
        elif q_var == 'node_q_b':
            var_mask = filter_q_b(swot_dict_arrays['node']['node_q_b'], q_val)
        elif q_var == 'xovr_cal_q':
            var_mask = ~np.isin(swot_dict_arrays['node']['xovr_cal_q'], q_val)
        elif q_var == 'wse_r_u_max':
            var_mask = np.greater(swot_dict_arrays['node']['wse_r_u'], q_val).filled(np.nan)
        else:
            logging.info(f'WARNING : unable to filter node using var {q_var}, not found in SWOT file or empty ... skip')
            var_mask = np.zeros((swot_dict_arrays['nx'], swot_dict_arrays['nt']))
        node_level_values_to_remove_mask = np.logical_or(node_level_values_to_remove_mask, var_mask)
        logging.info(f'Filtering SWOT Obs node with var {q_var} with values {q_val} : {np.sum(var_mask)} values to remove, {np.sum(~node_level_values_to_remove_mask)} left')
    for q_var, q_val in val_swot_q_flag['reach'].items():
        if q_var == 'obs_frac_n_min':
            var_mask = np.less(swot_dict_arrays['reach']['obs_frac_n'], q_val).filled(np.nan)
        elif q_var == 'ice_clim_f':
            var_mask = ~np.isin(swot_dict_arrays['reach']['ice_clim_f'], q_val)
        elif q_var == 'partial_f_max':
            var_mask = np.greater(swot_dict_arrays['reach']['partial_f'], q_val).filled(np.nan)
        elif q_var == 'slope_r_u_max':
            var_mask = np.greater(swot_dict_arrays['reach']['slope_r_u'], q_val).filled(np.nan)
        elif q_var == 'slope2_r_u_max':
            var_mask = np.greater(swot_dict_arrays['reach']['slope2_r_u'], q_val).filled(np.nan)
        elif q_var == 'reach_q_b':
            var_mask = filter_q_b(swot_dict_arrays['reach']['reach_q_b'], q_val)
        elif q_var == 'dark_frac_max':
            var_mask = np.greater(swot_dict_arrays['reach']['dark_frac'], q_val).filled(np.nan)
        elif q_var == 'width_u_max':
            var_mask = np.greater(swot_dict_arrays['reach']['width_u'], q_val).filled(np.nan)
        elif q_var == 'wse_r_u_max':
            var_mask = np.greater(swot_dict_arrays['reach']['wse_r_u'], q_val).filled(np.nan)
        elif q_var == 'wse_u_max':
            var_mask = np.greater(swot_dict_arrays['reach']['wse_u'], q_val).filled(np.nan)
        elif q_var == 'xtrk_dist_max':
            var_mask = np.greater(np.abs(swot_dict_arrays['reach']['xtrk_dist']), q_val).filled(np.nan)
        elif q_var == 'xtrk_dist_min':
            var_mask = np.less(np.abs(swot_dict_arrays['reach']['xtrk_dist']), q_val).filled(np.nan)
        elif q_var == 'xovr_cal_q':
            var_mask = ~np.isin(swot_dict_arrays['reach']['xovr_cal_q'], q_val)
        elif q_var == 'reach_length_min':
            if swot_dict_arrays['reach'][q_var] < q_val:
                var_mask = np.tile(np.nan, (1, swot_dict_arrays['nt']))
            else:
                var_mask = np.tile(False, (1, swot_dict_arrays['nt']))
        elif q_var == 'reach_width_min':
            if swot_dict_arrays['reach'][q_var] < q_val:
                var_mask = np.tile(np.nan, (1, swot_dict_arrays['nt']))
            else:
                var_mask = np.tile(False, (1, swot_dict_arrays['nt']))
        elif q_var == 'reach_slope_min':
            if swot_dict_arrays['reach'][q_var] < q_val:
                var_mask = np.tile(np.nan, (1, swot_dict_arrays['nt']))
            else:
                var_mask = np.tile(False, (1, swot_dict_arrays['nt']))
        else:
            logging.info(f'WARNING : unable to filter reach using var {q_var}, not found in SWOT file or empty ... skip')
            var_mask = np.zeros((1, swot_dict_arrays['nt']))
        '\n        elif q_var == "tukey":\n            tukey_number = q_val\n            index_non_nan = np.where(check_na_vector(reach_w)==False)\n            tukey_res = remove_outliers_tukey(reach_w[index_non_nan],tukey_number)\n            reach_w[index_non_nan] = tukey_res\n        '
        var_mask = np.tile(var_mask, (swot_dict_arrays['nx'], 1))
        node_level_values_to_remove_mask = np.logical_or(node_level_values_to_remove_mask, var_mask)
        logging.info(f'Filtering SWOT Obs reach with var {q_var} with values {q_val} : {np.sum(var_mask)} values to remove, {np.sum(~node_level_values_to_remove_mask)} left')
    return node_level_values_to_remove_mask

def filter_dict_using_mask(swot_dict_arrays, mask_array):
    out_swot_dict_arrays = {}
    for k in ['node', 'reach']:
        out_swot_dict_arrays[k] = {}
        for var, v_array in swot_dict_arrays[k].items():
            if var in ['node_id', 'time']:
                out_swot_dict_arrays[k][var] = v_array
            else:
                out_swot_dict_arrays[k][var] = filter_array_using_mask(v_array, mask_array[k])
    return out_swot_dict_arrays

def filter_array_using_mask(value_array, mask_array):
    filtered_value_array = np.where(mask_array, value_array, np.nan)
    return filtered_value_array

def get_flag_dict_from_config(config):
    val_swot_q_flag_node = {}
    val_swot_q_flag_reach = {}
    for param, param_value in config.items():
        if param.startswith('node_level_'):
            if param.endswith('_f') or param.endswith('_q') or param.endswith('_q_b'):
                val_swot_q_flag_node[param.replace('node_level_', '')] = [int(p) for p in param_value.split(',')]
            else:
                val_swot_q_flag_node[param.replace('node_level_', '')] = float(param_value)
        elif param.startswith('reach_level_'):
            if param.endswith('_f') or param.endswith('_q') or param.endswith('_q_b'):
                if isinstance(param_value, int):
                    val_swot_q_flag_reach[param.replace('reach_level_', '')] = float(param_value)
                elif isinstance(param_value, str):
                    val_swot_q_flag_reach[param.replace('reach_level_', '')] = [int(p) for p in param_value.split(',')]
            else:
                val_swot_q_flag_reach[param.replace('reach_level_', '')] = float(param_value)
    val_swot_q_flag = {'node': val_swot_q_flag_node, 'reach': val_swot_q_flag_reach}
    return val_swot_q_flag

def get_var_description_from_riversp_xml_file(riversp_xml_file_path, var):
    tree = ET.parse(riversp_xml_file_path)
    root = tree.getroot()
    var_description = {}
    for i in root.findall(f'.//{var}'):
        for j in i.iter():
            var_description[j.tag] = j.text
    return var_description

def get_list_of_flags_from_decimal(dec_num):
    bit_positions = []
    position = 0
    while dec_num > 0:
        if dec_num & 1:
            bit_positions.append(position)
        dec_num >>= 1
        position += 1
    bit_positions.sort(reverse=True)
    return bit_positions

def filter_q_b(q_b_array, q_b_target):
    q_b_target.sort(reverse=True)
    q_b_target_dec = sum((1 << i for i in q_b_target))
    mask_array = np.zeros(q_b_array.shape).astype('int')
    q_b_array_unique_values, counts = np.unique(q_b_array, return_counts=True)
    for q_b_decimal_value in q_b_array_unique_values.astype(int):
        q_b_value = get_list_of_flags_from_decimal(q_b_decimal_value)
        if len(set(q_b_target) & set(q_b_value)) > 0:
            logging.info(f'Discards {np.sum(q_b_array == q_b_decimal_value)} values with q_b flags {q_b_value} ; {q_b_target}')
            logging.debug(f'Decimal value : {q_b_decimal_value}; decimal target {q_b_target_dec}')
            logging.debug(f'Binary value : {bin(q_b_decimal_value)}; binary target {bin(q_b_target_dec)}')
            mask_array[q_b_array == q_b_decimal_value] = 1
    return mask_array

def main():
    q_b_array = np.array([40962.0, 40970.0, 524290.0, 524298.0, 524302.0, 526346.0, 526350.0, 540682.0, 542730.0, 557066.0, 559114.0, 565248.0, 565254.0, 565258.0, 567296.0, 567298.0, 567302.0, 567304.0, 567306.0, 573450.0, 575498.0, 576522.0, 804874.0, 469762048.0, 503349248.0])
    vals = [9, 10, 11, 22, 23, 24, 25, 26, 27, 28]
    mask = filter_q_b(q_b_array, vals)
if __name__ == '__main__':
    main()