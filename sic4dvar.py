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
            All functions not mentioned below.
          Hind OUBANAS, hind.oubanas@inrae.fr
            All functions not mentioned below.
          Nikki TEBALDI, ntebaldi@umass.edu
            get_input_data(),
            write_output(),
            get_reachids(),
            get_reach_dataset(),
            part of main()
        Dylan QUITTARD, dylan.quittard@inrae.fr
            All functions not mentioned above.
        Cécile Cazals, cecile.cazals@cs-soprasteria.com
            logger
            
Description:
    Version of algo31 and algo5 combined into algo315 (sic4dvar) which can 
    run on Confluence.
    NOTE this code is for use only for computational resource testing. 
    Numerous modification need to be made to run accurately on confluence.
    See the TODO list at the top of this file.
"""
import argparse
import logging
import sys
import time
from pathlib import Path
from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)
import sic4dvar_params as params
from lib.lib_config import read_config
from lib.lib_log import append_to_principal_log, call_error_message
from sic4dvar_functions.sic4dvar_helper_functions import get_run_type
from sic4dvar_functions.sic4dvar_runs import sic4dvar_run

def get_explicit_args(parser):
    """Returns a set of argument dest names that were explicitly set via the command line."""
    explicit = set()
    for action in parser._actions:
        if any((opt in sys.argv for opt in action.option_strings)):
            explicit.add(action.dest)
    return explicit

def main():
    """Main method to read in data and write out data."""
    parser = argparse.ArgumentParser(description='Sic4dvar algorithm.')
    parser.add_argument('--config_file_path', help='json file name to run.', default=Path(__file__).parent / 'sic4dvar_param.ini', type=Path)
    parser.add_argument('-r', '--json_path', help='json file name to run.', type=Path)
    parser.add_argument('--output_dir', help='Output directory', type=Path)
    parser.add_argument('--log_dir', help='Log directory', type=Path)
    parser.add_argument('--swot_dir', help='Input swot directory', type=Path)
    parser.add_argument('--sos_dir', help='SWORD Of Science dir', type=Path)
    parser.add_argument('--sword_dir', help='SWORD dir', type=Path)
    parser.add_argument('--input_dir', help='Input directory with swot, sos, and sword directories inside', type=Path)
    parser.add_argument('-p', '--flag_parallel', action='store_true', help='Parallel processing')
    parser.add_argument('-l', '--log_level', help='Set log level (name only: DEBUG, INFO, WARNING, ERROR, CRITICAL)', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], type=str)
    parser.add_argument('-i', '--index', help='json file index for reach/set', type=int)
    args = parser.parse_args()
    if params.replace_config:
        args.config_file_path = Path(params.config_file_path)
    param_dict = read_config(args.config_file_path)
    explicit_args = get_explicit_args(parser)
    args_dict = vars(args)
    for key in explicit_args:
        param_dict[key] = args_dict[key]
    if param_dict['log_level'].upper() not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        print(f"Invalid log level '{param_dict['log_level']}' : set to INFO")
        param_dict['log_level'] = 'INFO'
    param_dict['log_path'] = param_dict['log_dir'].joinpath('sic4dvar.log')
    if not param_dict['output_dir'].exists():
        param_dict['output_dir'].mkdir(parents=True, exist_ok=True)
    append_to_principal_log(param_dict, f"Invalid log level '{param_dict['log_level']}' : set to INFO")
    append_to_principal_log(param_dict, 'Running SIC4DVAR low cost')
    if param_dict['aws'] and 'json_path' not in param_dict:
        param_dict['json_path'] = Path(param_dict['input_dir'], 'reaches.json')
        param_dict['sos_dir'] = param_dict['input_dir'].joinpath('sos')
        param_dict['swot_dir'] = param_dict['input_dir'].joinpath('swot')
        param_dict['sword_dir'] = param_dict['input_dir'].joinpath('sword')
    elif param_dict['aws']:
        param_dict['json_path'] = Path(param_dict['input_dir'], param_dict['json_path'])
        param_dict['sos_dir'] = param_dict['input_dir'].joinpath('sos')
        param_dict['swot_dir'] = param_dict['input_dir'].joinpath('swot')
        param_dict['sword_dir'] = param_dict['input_dir'].joinpath('sword')
    param_dict['run_type'] = get_run_type(param_dict['json_path'])
    append_to_principal_log(param_dict, f'A {param_dict['run_type']} is detected since json file')
    if param_dict['run_type'] in ['seq', 'set']:
        sic4dvar_run(param_dict)
    else:
        append_to_principal_log(param_dict, f'Unknown run type {param_dict['run_type']}')
        pass
if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('Elapsed time:', end - start, 'seconds')