from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)
import pandas as pd
import scipy
import logging
import numpy as np
import sic4dvar_params as params
import sic4dvar_functions.sic4dvar_calculations as calc
from sic4dvar_functions.sic4dvar_helper_functions import get_weighted_q_data
from lib.lib_dates import get_swot_dates
from lib.lib_verif import check_na
from lib.lib_indicators import integrated_mean, integrated_variance

def replace_prior(sic4dvar_dict):
    flag_qwbm = True
    if (check_na(sic4dvar_dict['input_data']['reach_qwbm']) or sic4dvar_dict['input_data']['reach_qwbm'] < 1.0) and sic4dvar_dict['param_dict']['q_prior_from_stations']:
        logging.warning('gauge prior value invalid/too small.')
        flag_qwbm = False
        logging.info(f'INFO: model value invalid for station {sic4dvar_dict['input_data']['reach_id']}.')
    if (check_na(sic4dvar_dict['input_data']['reach_qwbm']) or sic4dvar_dict['input_data']['reach_qwbm'] < 1.0) and (not sic4dvar_dict['param_dict']['q_prior_from_stations']) and (not params.activate_facc):
        logging.warning('qwbm value invalid/too small.')
        logging.info('INFO: no option to replace model value.')
        flag_qwbm = False
    if sic4dvar_dict['param_dict']['override_q_prior']:
        sic4dvar_dict['input_data']['reach_qwbm'] = np.ma.masked_values(float(sic4dvar_dict['param_dict']['q_prior_value']), value=-9999.0)
        flag_qwbm = True
    if sic4dvar_dict['data_is_useable'] and sic4dvar_dict['param_dict']['q_monthly_mean']:
        times = sic4dvar_dict['input_data']['reach_t']
        count = 0
        for i in range(0, len(sic4dvar_dict['input_data']['q_monthly_mean'])):
            if check_na(sic4dvar_dict['input_data']['q_monthly_mean'][i]):
                count = count + 1
        if count == len(sic4dvar_dict['input_data']['q_monthly_mean']):
            logging.info('INFO: tried to replaced qwbm value with q monthly mean computation')
            logging.info('But all q monthly mean data was empty !')
        else:
            dates = get_swot_dates(sic4dvar_dict['filtered_data']['node_t'])
            masked_data = np.ma.masked_values(np.array([get_weighted_q_data(dates, sic4dvar_dict['input_data']['q_monthly_mean'])]), value=-9999.0)
            sic4dvar_dict['input_data']['reach_qwbm'] = masked_data
            logging.info('INFO: replaced qwbm value with monthly mean computation')
            logging.info(f'qwbm value: {sic4dvar_dict['input_data']['reach_qwbm']}')
    if sic4dvar_dict['input_data']['create_series_from_monthly_q']:
        if 'q_monthly_series' in sic4dvar_dict['input_data']:
            check_na_vector = np.vectorize(check_na)
            if np.all(check_na_vector(sic4dvar_dict['input_data']['q_monthly_mean'])):
                logging.warning('All monthly Q data is NaN.')
                flag_qwbm = False
            else:
                mean = integrated_mean(np.array(sic4dvar_dict['input_data']['q_monthly_series'])[sic4dvar_dict['list_to_keep']], sic4dvar_dict['filtered_data']['reach_t'])
                std = integrated_variance(np.array(sic4dvar_dict['input_data']['q_monthly_series'])[sic4dvar_dict['list_to_keep']], sic4dvar_dict['filtered_data']['reach_t'], mean)
                flag_qwbm = True
                masked_data = np.ma.masked_values(np.array([mean]), value=-9999.0)
                sic4dvar_dict['input_data']['reach_qwbm'] = masked_data
                sic4dvar_dict['input_data']['series_mean'] = mean
                masked_data2 = np.ma.masked_values(np.array([std]), value=-9999.0)
                sic4dvar_dict['input_data']['reach_qsdev'] = masked_data2
                sic4dvar_dict['input_data']['series_std'] = std
                sic4dvar_dict['input_data']['series_cv'] = std / mean
    if (check_na(sic4dvar_dict['input_data']['reach_qwbm']) or sic4dvar_dict['input_data']['reach_qwbm'] < 1.0) and (not sic4dvar_dict['param_dict']['q_prior_from_stations']) and params.activate_facc:
        logging.warning('qwbm value invalid/too small.')
        logging.info('INFO: trying to replace qwbm value with facc * specific discharge.')
        flag_qwbm = False
        if not check_na(sic4dvar_dict['input_data']['facc']) or sic4dvar_dict['input_data']['facc'] > 1.0:
            sic4dvar_dict['input_data']['reach_qwbm'] = sic4dvar_dict['input_data']['facc'] * 10 / 1000
            logging.info('INFO: replaced qwbm value with facc * specific discharge.')
            logging.info(f'facc * SD value: {sic4dvar_dict['input_data']['reach_qwbm']}')
            flag_qwbm = True
    if flag_qwbm == True:
        logging.info('QWBM suitable to use.')
    return (sic4dvar_dict, flag_qwbm)