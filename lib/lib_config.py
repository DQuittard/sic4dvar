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
import configparser
import pathlib
import re

def read_config(config_file):
    out_config = {}
    config = configparser.ConfigParser()
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8', errors='replace') as f:
            config.read_file(f)
        for section in config.sections():
            for var, value in config.items(section):
                if section == 'PATH' or section == 'DATABASE':
                    out_config[var] = pathlib.Path(value)
                elif value.lower() == 'true':
                    out_config[var] = True
                elif value.lower() == 'false':
                    out_config[var] = False
                elif value.isdigit():
                    out_config[var] = int(value)
                elif re.match('^-?\\d+\\.\\d+$', value):
                    out_config[var] = value
                else:
                    out_config[var] = value
    else:
        print(f'Config file {config_file} not found ...')
    return out_config

def try_convert(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value