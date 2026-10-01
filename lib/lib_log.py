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
import portalocker
import os
log_messages = {'1': '', '2': '', '102': 'JSON file not found.', '103': 'Run on only one reach. Stopping.', '104': 'No SOS file {sos_file} found for reach of reach id: {reach_id}', '105': 'Reaches ending in something different from 1 are not processed in set mode. Skipping reach {reach_id}', '502': 'After filtering dates, no time instant left to compute mean of q prior from stations.', '503': "No time instant available on the reach to use stations' data.", '504': 'No SWORD file {sword_file} found for reach of reach id: {reach_id}', '505': 'No SWOT file {swot_file} found for reach of reach id: {reach_id}', '506': 'Mean of q prior from stations is <= 0. Exiting.', '507': 'QWBM < 1, discarded.', '999': 'Error message not found.'}

def call_error_message(error_code):
    try:
        message = log_messages[str(error_code)]
    except:
        message = log_messages['999']
    return message

def set_logger(param_dict, filename=None):
    logFormatter = logging.Formatter('%(asctime)s [%(levelname)-4.4s]  %(message)s')
    logger = logging.getLogger()
    logger.setLevel(level=param_dict['log_level'])
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    if param_dict['log_level'] in ['DEBUG', 'INFO']:
        os.makedirs(filename.parent, exist_ok=True)
        fileHandler = logging.FileHandler(filename, mode='w')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

def close_logger(param_dict):
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)

def append_to_principal_log(param_dict, message):
    if param_dict['log_level'] in ['DEBUG', 'INFO']:
        log_path = param_dict['log_path']
        os.makedirs(log_path.parent, exist_ok=True)
        if not log_path.parent.exists():
            os.makedirs(log_path.parent, exist_ok=True)
        with portalocker.Lock(log_path, mode='a+', timeout=5) as f:
            f.seek(0)
            if f.read(1) == '':
                f.write('Configuration is : \n')
                print('Configuration is : \n')
                for k, v in param_dict.items():
                    f.write(f'     {k} : {v}\n')
                    print(f'     {k} : {v}')
            f.write(message + '\n')