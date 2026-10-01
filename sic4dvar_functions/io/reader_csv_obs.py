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

Last modified on February 23rd 2024 at 18:00
by @Isadora Silva

@authors: Isadora Silva
"""
import pathlib
from datetime import datetime
from typing import Literal, Tuple
import numpy as np
import pandas as pd
from sic4dvar_classes.sic4dvar_0_defaults import SIC4DVarLowCostDefaults
from sic4dvar_low_cost.sic4dvar_functions.helpers.helpers_arrays import array_as_row_vector, array_as_col_vector, datetime_array_set_to_freq_and_filter
from sic4dvar_low_cost.sic4dvar_functions.T14 import D
from sic4dvar_low_cost.sic4dvar_functions.io.reader_sword import get_vars_from_sword_file

def get_array_dict_from_csv_files(reach_ids: Tuple[str, ...], node_z_file_pattern: str | pathlib.PurePath | None=None, node_z_u_file_pattern: str | pathlib.PurePath | None=None, node_w_file_pattern: str | pathlib.PurePath | None=None, node_w_u_file_pattern: str | pathlib.PurePath | None=None, reach_data_file_pattern: str | pathlib.PurePath | None=None, dist_col: str | None=None, node_id_col: str | None=None, reach_time_col: str | None=None, reach_s_col: str | None=None, reach_w_col: str | None=None, reach_da_col: str | None=None, miss_node_in_sword: str | None=None, no_data_value: float | str | int | None=None, dist_dif_obs: int | None=None, sword_file_path: str | pathlib.PurePath | None=None, x_ref: Literal['node_length', 'dist_out']='node_length', add_facc: bool=False, ref_datetime: datetime=SIC4DVarLowCostDefaults().def_ref_datetime, freq_datetime: str=SIC4DVarLowCostDefaults().def_freq_datetime, dup_datetime: Literal['drop', 'raise']='raise', start_datetime: datetime | float | int | None=None, end_datetime: datetime | float | int | None=None, clean_run: bool=False, debug_mode: bool=False):
    """
    One csv file per reach per node variable.
    In node variables (w) and (z): columns represent time instances, rows represent distances/node_ids. The distance
     can be found with the column dist_col, which is set as the index.
    Time must be in seconds from reference!

    One csv file per reach variables blended into one file.
    Columns are reach_s_col, reach_w_col, reach_da_col, rows are time instances. Time instances must match the ones
     of node elevation file.

    Raise KeyError in case node_w file for the same reach does not match the node_z_file (same time and dists).

    ATTENTION: Output is not sorted!

    Parameters
    ----------
    reach_ids : Tuple[str | int, ...]
        The reach ids to that are present in the csv file name.
    node_z_file_pattern : str | pathlib.PurePath
        the pattern of the file that contains the node water surface elevations.
         It is formatted according to reach_ids.
    node_z_u_file_pattern : str | pathlib.PurePath | None
        the pattern of the file that contains the node water surface elevation uncertainties.
         It is formatted according to reach_ids.
    node_w_file_pattern : str | pathlib.PurePath | None
        the pattern of the file that contains the node widths.
         It is formatted according to reach_ids.
    node_w_u_file_pattern : str | pathlib.PurePath | None
        the pattern of the file that contains the node width uncertainties.
         It is formatted according to reach_ids.
    reach_data_file_pattern : str | pathlib.PurePath | None
        the pattern of the file that contains the reach variables (da, s, w)
    dist_col : str | None
        the name of the column that contains the distance. If not defined, extract it from the SWORD file.
    node_id_col : str | None
        the name of the column that contains the node_ids.
         Must be defined if distance is extracted from the SWORD file.
    reach_time_col : str | None
        in the reach file, the name of the column that contains the time
    reach_s_col : str | None
        in the reach file, the name of the column that contains the slope
    reach_w_col : str | None
        in the reach file, the name of the column that contains the width
    reach_da_col : str | None
        in the reach file, the name of the column that contains the chagne in area
    miss_node_in_sword: str | None
        What do do in case a node exists in the csv files but not in SWORD. If None, raise KeyError, else, linear
         interpolate the distance.
    no_data_value : float | str | int | None
        the no data value in the csv file
    dist_dif_obs : int | None
        distance difference between the observations so they are considered at the same distance when loading
         the data
    sword_file_path : str | pathlib.PurePath | None
        The path to the SWORD netCDF file
    x_ref : Literal["node_length", "dist_out"]
        The reference for the distance.
         Due to inconsistencies with the SWORD, it is recommended to use "node_length".
    add_facc : bool
        whether to load facc from SWORD
    ref_datetime : datetime
        The reference datetime to convert between dates to seconds from reference.
    freq_datetime : datetime
        The time of observations is approximated every freq when loading the data.
        Multiplier and str "D", "h", "min", "s". E.g.: "3h", "D". Set as "" to skip this.
        https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    dup_datetime: Literal["drop", "raise"]
        What to do when the data is assigned to the same datetime. Either "drop", "raise": "drop" drops all
         occurrences except the first one; "raise" raises IndexError.
    start_datetime: datetime | float | int | None
        start datetime for filtering the data (>=). Either as datetime or as seconds from reference (float or int).
    end_datetime: datetime | float | int | None
        end datetime for filtering the data (<=). Either as datetime or as seconds from reference (float or int).
    clean_run : bool
        Whether to print statements while running this class
    debug_mode : bool
        Whether to print debug statements while running this class
    """
    if clean_run:
        debug_mode = False
    if debug_mode:
        clean_run = False
    if all([i_ is None for i_ in [node_z_file_pattern, node_w_file_pattern]]):
        raise TypeError('at least one of node_z_file_pattern, node_w_file_pattern must be specified')
    reach_ids = [int(i) for i in reach_ids]
    if len(reach_ids) == 0:
        raise TypeError('must specify at lest one reach_id')
    if dist_dif_obs is None:
        dist_dif_obs = SIC4DVarLowCostDefaults().def_dist_dif_obs
    if dist_col is None and node_id_col is None:
        raise TypeError('must define one of node_id_col or dist_col')
    if sword_file_path is not None:
        if node_id_col is None:
            raise TypeError('if sword_file_path is defined, node_id_col must be defined')
        sword_file_path = pathlib.Path(sword_file_path)
        if not sword_file_path.exists():
            raise FileNotFoundError(f'sword_file_path {sword_file_path} does not exist')
    else:
        add_facc = False
    df_node_list_var, df_reach_list_var, path_dict = ([], [], {})
    if node_z_file_pattern is not None:
        path_dict['node_z'] = str(node_z_file_pattern)
        df_node_list_var.append('node_z')
    if node_w_file_pattern is not None:
        path_dict['node_w'] = str(node_w_file_pattern)
        df_node_list_var.append('node_w')
    if node_z_u_file_pattern is not None:
        path_dict['node_z_u'] = str(node_z_u_file_pattern)
        df_node_list_var.append('node_z_u')
    if node_w_u_file_pattern is not None:
        path_dict['node_w_u'] = str(node_w_u_file_pattern)
        df_node_list_var.append('node_w_u')
    if reach_data_file_pattern is not None and np.any([i_ is not None for i_ in [reach_s_col, reach_w_col, reach_da_col]]):
        if reach_time_col is None:
            raise TypeError('to extract the time form reach csv, must specify reach_time_col')
        path_dict['reach_data'] = str(reach_data_file_pattern)
        if reach_s_col is not None:
            df_reach_list_var.append('reach_s')
        if np.all([i_ is not None for i_ in [reach_s_col, reach_w_col, reach_da_col]]):
            df_reach_list_var.append('reach_w')
            df_reach_list_var.append('reach_da')
    df_dict_arrays = {k_: [] for k_ in df_node_list_var + df_reach_list_var}
    df_dict_arrays['t'] = []
    df_dict_arrays['x'] = []
    df_dict_arrays['node_id'] = []
    for n_reach, reach_id in enumerate(reach_ids):
        for n_var, node_var in enumerate(df_node_list_var):
            f_pat = path_dict[node_var]
            df_var = pd.read_csv(f_pat.format(reach_id))
            df_var = df_var.astype(np.float64)
            if no_data_value is not None:
                df_var[df_var == no_data_value] = pd.NA
            dist_array = np.empty(0) if dist_col is None else array_as_col_vector(np.round(df_var[dist_col].to_numpy(dtype=np.float64), 0).astype(np.int64))
            node_id_array = np.empty(0) if node_id_col is None else array_as_col_vector(df_var[node_id_col].to_numpy(dtype=np.int64))
            df_var = df_var[[c_ for c_ in df_var.columns if c_ not in [node_id_col, dist_col]]]
            datetime_sec_df_array = np.round(df_var.columns.to_numpy(dtype=np.float64), 0).astype(np.int64)
            df_var.columns = datetime_sec_df_array
            datetime_sec_df_array = np.sort(datetime_sec_df_array)
            df_var = df_var[datetime_sec_df_array]
            if any([start_datetime is not None, end_datetime is not None, freq_datetime]):
                cols_mask_bool, _, datetime_sec_df_array = datetime_array_set_to_freq_and_filter(data_dt=datetime_sec_df_array, ref_datetime=ref_datetime, freq_datetime=freq_datetime, duplicates=dup_datetime, start_datetime=start_datetime, end_datetime=end_datetime)
                df_var = df_var[df_var.columns[cols_mask_bool]]
                df_var.columns = datetime_sec_df_array
            time_array = array_as_row_vector(datetime_sec_df_array)
            if n_var == 0:
                df_dict_arrays['t'].append(time_array)
                df_dict_arrays['x'].append(dist_array)
                df_dict_arrays['node_id'].append(node_id_array)
            else:
                error_msg = f'reach {reach_id} variable {node_var},'
                if df_dict_arrays['t'][n_reach].shape != time_array.shape:
                    raise IndexError(f'{error_msg} time array shape mismatch')
                if not np.allclose(df_dict_arrays['t'][n_reach], time_array, rtol=0, atol=0):
                    raise ValueError(f'{error_msg} time array value mismatch')
                if df_dict_arrays['x'][n_reach].shape != dist_array.shape:
                    raise IndexError(f'{error_msg} dist array shape mismatch')
                if not np.allclose(df_dict_arrays['x'][n_reach], dist_array, rtol=0.0, atol=dist_dif_obs):
                    raise ValueError(f'{error_msg} dist array value mismatch')
                if df_dict_arrays['node_id'][n_reach].shape != node_id_array.shape:
                    raise IndexError(f'{error_msg} node id array shape mismatch')
                if not np.all(np.equal(df_dict_arrays['node_id'][n_reach], node_id_array)):
                    raise ValueError(f'{error_msg} node id array value mismatch')
            df_dict_arrays[node_var].append(df_var.to_numpy(dtype=np.float32))
        if 'reach_data' in path_dict.keys():
            f_pat = path_dict['reach_data']
            df_reach = pd.read_csv(f_pat.format(reach_id))
            if no_data_value is not None:
                df_reach.loc[df_reach == no_data_value] = pd.NA
            df_reach = df_reach.astype(np.float64)
            df_reach.sort_values(reach_time_col, inplace=True)
            datetime_sec_df_array = np.round(df_reach[reach_time_col].to_numpy(dtype=np.float64).flatten(), 0).astype(np.int64)
            if any([start_datetime is not None, end_datetime is not None, freq_datetime]):
                rows_mask_bool, _, datetime_sec_df_array = datetime_array_set_to_freq_and_filter(data_dt=datetime_sec_df_array, ref_datetime=ref_datetime, freq_datetime=freq_datetime, duplicates=dup_datetime, start_datetime=start_datetime, end_datetime=end_datetime)
                df_reach = df_reach.loc[df_reach.index[rows_mask_bool]]
            time_array = array_as_row_vector(datetime_sec_df_array)
            df_reach = df_reach[[c_ for c_ in df_reach.columns if c_ != reach_time_col]]
            error_msg = f'reach {reach_id} reach variables,'
            if df_dict_arrays['t'][n_reach].shape != time_array.shape:
                raise IndexError(f'{error_msg} time array shape mismatch')
            if not np.allclose(df_dict_arrays['t'][n_reach], time_array, rtol=0, atol=0):
                raise ValueError(f'{error_msg} time array value mismatch')
            for reach_var in df_reach_list_var:
                df_dict_arrays[reach_var].append(array_as_row_vector(df_reach[reach_var].to_numpy(dtype=np.float32)))
    df_dict_arrays['reach_id'] = [np.full((1, 1), fill_value=i_, dtype=np.int64) for i_ in reach_ids]
    if sword_file_path is not None and node_id_col is not None:
        sword_dict = get_vars_from_sword_file(reach_ids=reach_ids, sword_file_path=sword_file_path, node_vars=('node_length', 'dist_out', 'node_id'), reach_vars=('facc',) if add_facc else (), x_ref=x_ref, clean_run=clean_run)
        sword_dict['nodes']['node_id'] = sword_dict['nodes']['node_id'].astype(np.int64)
        for reach_n in range(len(df_dict_arrays['reach_id'])):
            df_dict_arrays['x'][reach_n] = np.full(df_dict_arrays['node_id'][reach_n].shape, fill_value=np.nan, dtype=np.float64)
            for node_idx_in_df, node_n in enumerate(df_dict_arrays['node_id'][reach_n][:, 0]):
                try:
                    node_idx_in_sword = np.nonzero(sword_dict['nodes']['node_id'] == node_n)[0][0]
                except IndexError:
                    error_msg = f'could not find node {node_n} in SWORD for reach ' + str(df_dict_arrays['reach_id'][reach_n][0, 0])
                    if miss_node_in_sword is None or any([i_ in miss_node_in_sword for i_ in ['raise', 'error']]):
                        raise KeyError(error_msg)
                    if not clean_run:
                        print(error_msg)
                else:
                    df_dict_arrays['x'][reach_n][node_idx_in_df] = sword_dict['nodes']['dist_out'][node_idx_in_sword]
            if np.any(np.isnan(df_dict_arrays['x'][reach_n])):
                df_dict_arrays['x'][reach_n] = D(values_in_array=df_dict_arrays['x'][reach_n], base_in_array=np.array(range(0, len(df_dict_arrays['x'][reach_n]))), limits='linear', check_nan=False)
        for reach_n in range(len(df_dict_arrays['reach_id'])):
            df_dict_arrays['x'][reach_n] = np.nanmax(sword_dict['nodes']['dist_out']) - df_dict_arrays['x'][reach_n]
    else:
        sword_dict = dict()
    if df_dict_arrays['node_id'][0].size == 0:
        df_dict_arrays.pop('node_id')
    if df_dict_arrays['x'][0].size == 0:
        df_dict_arrays.pop('x')
    if add_facc:
        df_dict_arrays['reach_facc'] = [np.full((1, 1), fill_value=i_, dtype=np.float32) for i_ in sword_dict['reaches']['facc']]
    return df_dict_arrays
if __name__ == '__main__':
    from pathlib import Path
    base_path = Path('C:\\Users\\isadora.rezende\\PhD')
    swot_sim_path = base_path / 'Discharge_paper' / 'Po' / 'SWOT_sim' / 'csvs' / 'observations'
    test_output = get_array_dict_from_csv_files(reach_ids=(21406100071, 21406100031, 21406100041), node_z_file_pattern=swot_sim_path / '{}_z_SWOT_sim.csv', node_w_file_pattern=swot_sim_path / '{}_w_SWOT_sim.csv', node_id_col='node_id', miss_node_in_sword='linear', no_data_value=-9999.0, sword_file_path=base_path / 'Datasets' / 'SWORD' / 'v15' / 'netcdf' / 'eu_sword_v15.nc', start_datetime=datetime(2008, 5, 1), end_datetime=datetime(2009, 6, 30))
    for k, v in test_output.items():
        print(k, [v_i.shape for v_i in v])