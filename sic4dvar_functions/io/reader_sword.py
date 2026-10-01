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

Created on September 18th 2023 at 19:00
by @Isadora Silva

Last modified on February 21st 2024 at 18:45
by @Isadora Silva

@authors: Isadora Silva
"""
import pathlib
from typing import Literal, Tuple
import netCDF4 as nc4
import numpy as np
import scipy
from sic4dvar_functions.helpers.helpers_arrays import masked_array_to_nan_array

def get_vars_from_sword_file(reach_ids: Tuple[str | int, ...], sword_file_path: str | pathlib.PurePath, node_vars: Tuple[str, ...]=(), reach_vars: Tuple[str, ...]=(), x_ref: Literal['node_length', 'dist_out']='node_length', add_reach_dist_out: bool=False, clean_run: bool=False) -> dict:
    """
    Read the node_vars and the reach_vars from SWORD netcdf file and return a dictionary with the unmasked arrays.

    If x_ref is the node_length, the values of dist_out in the dictionary do not match the SWORD dist_out. They
     represent an accumulated sum of the node lengths corrected by the mode of the difference between SWORD dist_out
     and this accumulation (this makes sure dist_out is always increasing/decreasing).

    ATTENTION: Output is not sorted!

    Parameters
    ----------
    reach_ids : Tuple[str, ...]
        The reach ids to be read from the SWORD file
    sword_file_path : str | pathlib.PurePath
        The path to the SWORD netCDF file
    node_vars : Tuple[str, ...]
        The node variables to be loaded
    reach_vars : Tuple[str, ...]
        The reach variables to be loaded
    x_ref : Literal["node_length", "dist_out"]
        The reference for the distance. Due to inconsistencies with the SWORD, it is recommended to use "node_length".
    add_reach_dist_out : bool
        whether to add the distance for each reach to the output. Computed as the average dist_out from the nodes.
    clean_run : bool
        Whether to print statements while running this function

    Returns
    -------
    dict
        Dictionary with the unmasked arrays. Structure {"nodes": {node_dict}, "reaches": {reach_dict}}. Each inner dict
         contains an array whose elements are the unmasked arrays for the combined reach_ids.
    """
    msg = 'loading data from SWORD'
    if not clean_run:
        print(msg)
    if not node_vars and (not reach_vars):
        raise TypeError('must specify either node_vars or reach_vars')
    if reach_ids is not None:
        reach_ids = [int(i) for i in reach_ids]
    sword_file_path = pathlib.Path(sword_file_path)
    if not sword_file_path.exists():
        raise FileNotFoundError(f'sword_file_path {sword_file_path} does not exist')
    if x_ref not in ['node_length', 'dist_out']:
        raise TypeError('x_ref must be one of node_length or dist_out')
    sword_dict = {'nodes': {k_: [] for k_ in node_vars}, 'reaches': {k_: [] for k_ in reach_vars}}
    if 'node_length' in node_vars or 'dist_out' in node_vars or add_reach_dist_out:
        sword_dict['nodes']['dist_out'] = []
        sword_dict['nodes']['node_length'] = []
        if add_reach_dist_out:
            sword_dict['reaches']['dist_out'] = np.full(len(reach_ids), fill_value=np.nan, dtype=np.float64)
    with nc4.Dataset(sword_file_path) as sword_ds:
        sword_reach_ids = sword_ds['reaches']['reach_id'][:]
        if reach_ids is None:
            reach_ids = sword_reach_ids
        if len(sword_dict['reaches'].keys()) > 0:
            for r_i in reach_ids:
                sword_reach_index_array = (sword_reach_ids == r_i).nonzero()[0]
                for reach_var in reach_vars:
                    reach_var_array = sword_ds['reaches'][reach_var][sword_reach_index_array]
                    reach_var_array = masked_array_to_nan_array(reach_var_array)
                    sword_dict['reaches'][reach_var].append(reach_var_array[0])
            for reach_var in reach_vars:
                sword_dict['reaches'][reach_var] = np.array(sword_dict['reaches'][reach_var])
        if len(sword_dict['nodes'].keys()) > 0:
            sword_node_reach_ids = sword_ds['nodes']['reach_id'][:]
            for n_i, r_i in enumerate(reach_ids):
                sword_node_index_array = (sword_node_reach_ids == r_i).nonzero()[0]
                for node_var in sword_dict['nodes'].keys():
                    node_var_array = sword_ds['nodes'][node_var][sword_node_index_array]
                    node_var_array = masked_array_to_nan_array(node_var_array)
                    sword_dict['nodes'][node_var].append(node_var_array)
                if 'dist_out' in sword_dict['nodes'].keys() and x_ref.lower() != 'dist_out':
                    dist_out_array = sword_dict['nodes']['dist_out'][n_i]
                    node_length_array = sword_dict['nodes']['node_length'][n_i]
                    acc_node_len = np.cumsum(node_length_array)
                    dist_out_0 = scipy.stats.mode(np.abs(acc_node_len - dist_out_array), keepdims=False).mode
                    sword_dict['nodes']['dist_out'][n_i] = dist_out_0 + acc_node_len
                if add_reach_dist_out:
                    sword_dict['reaches']['dist_out'][n_i] = np.nanmean(sword_dict['nodes']['dist_out'][n_i])
            for node_var in sword_dict['nodes'].keys():
                sword_dict['nodes'][node_var] = np.concatenate(sword_dict['nodes'][node_var])
        if add_reach_dist_out:
            reach_vars = tuple(list(reach_vars) + ['dist_out'])
        out_dict = {'nodes': {k_: sword_dict['nodes'][k_] for k_ in node_vars}, 'reaches': {k_: sword_dict['reaches'][k_] for k_ in reach_vars}}
        if not clean_run:
            print(msg, 'done')
        return out_dict
if __name__ == '__main__':
    base_path = pathlib.Path('C:\\Users\\isadora.rezende\\PhD')
    sword_f = base_path / 'Datasets' / 'SWORD' / f'v15' / 'netcdf' / f'eu_sword_v15.nc'
    sword_dict = get_vars_from_sword_file(reach_ids=(21406100071, 21406100031, 21406100041), sword_file_path=sword_f, node_vars=('dist_out', 'node_id'), reach_vars=('facc', 'reach_id'), x_ref='node_length', add_reach_dist_out=True, clean_run=False)
    print(sword_dict)