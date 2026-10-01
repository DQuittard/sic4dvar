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
import numpy as np
from netCDF4 import Dataset
from lib.lib_netcdf import get_nc_variable_data

class Sword(object):

    def __init__(self, sword_path, sword_version='16'):
        self.sword = Dataset(sword_path)

    def get_node_id_list(self, reach_id_list):
        nodes_idx = np.isin(get_nc_variable_data(self.sword, 'nodes/reach_id').filled(np.nan), reach_id_list)
        node_id_list = get_nc_variable_data(self.sword, 'nodes/node_id').filled(np.nan)[nodes_idx].astype(int)
        return node_id_list

    def get_var_from_group(self, group, var_name):
        try:
            return self.sword[group][var_name][:]
        except:
            logging.warning(f'Variable {var_name} not found in SWORD file group {group}')

    def close(self):
        self.sword.close()

def compute_sword_path_from_cont(sword_path, cont):
    for file in sword_path.glob(f'{cont.lower()}_sword_v*.nc'):
        return file
    logging.warning(f'SWORD file not found in {sword_path} for continent {cont}')
    return None