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

Created on November 9th 2023 at 11:00
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
from sic4dvar_functions.helpers.helpers_arrays import datetime_array_set_to_freq_and_filter
from sic4dvar_functions.io.reader_sword import get_vars_from_sword_file
from sic4dvar_functions.utm_converter import load_utm_crs_from_latlon

def read_hydroweb_txt(hydroweb_txt_file_path: str | pathlib.PurePath, no_data_value: float | str=9999.999, ref_datetime: datetime=SIC4DVarLowCostDefaults().def_ref_datetime, freq_datetime: str=SIC4DVarLowCostDefaults().def_freq_datetime, dup_datetime: Literal['drop', 'raise']='raise', start_datetime: datetime | float | int | None=None, end_datetime: datetime | float | int | None=None) -> Tuple[pd.DataFrame, float, float]:
    """
    Parameters
    ----------
    hydroweb_txt_file_path : str | pathlib.PurePath
        The path to the csv file containing the data from a Hydroweb altimetry station
    no_data_value : float
        The value to be considered as NaN. Default is 9999.999.
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

    Returns
    -------
    Tuple[pd.DataFrame, float, float]
        (i) latitude of station
        (ii) longitude of station
        (iii) DataFrame with tabular data, index as datetime, columns: wse and wse_u
    """
    lat_s, lon_s = (np.nan, np.nan)
    df_s = pd.DataFrame()
    j = 0
    start_table = False
    col_descript = []
    with open(hydroweb_txt_file_path, mode='r') as fp:
        f_lines = fp.readlines()
        for row in f_lines:
            if 'LONGITUDE::' in row:
                lon_s = float(row.split(' ')[-1])
            if 'LATITUDE::' in row:
                lat_s = float(row.split(' ')[-1])
            if '#COL' in row:
                col_descript.append(row.split(' : ')[-1][:-1])
            if '####' in row:
                start_table = True
                continue
            if start_table:
                if df_s.columns.empty:
                    df_s = pd.DataFrame(columns=col_descript)
                row = row.split(' ')
                row = [i for i in row if i != ':']
                row = [i if i != str(no_data_value) else np.nan for i in row]
                df_s.loc[j] = row
                j += 1
    date_col, z_col, z_u_col = ('', '', '')
    for c in col_descript:
        if 'time(' in c.lower():
            continue
        elif 'date(' in c.lower():
            df_s[c] = pd.to_datetime(df_s[c] + ' ' + df_s[[c_i for c_i in col_descript if 'time(' in c_i.lower()][0]])
            date_col = c
            continue
        elif 'orthometric' in c.lower():
            z_col = c
        elif 'uncertainty' in c.lower():
            z_u_col = c
        df_s = df_s.astype({c: np.float32}, errors='ignore')
    df_s = df_s[[date_col, z_col, z_u_col]]
    df_s.rename(columns={date_col: 'date', z_col: 'wse', z_u_col: 'wse_u'}, inplace=True)
    df_s.set_index('date', inplace=True)
    if any([start_datetime is not None, end_datetime is not None, freq_datetime]):
        df_s_index_mask_bool, df_s_index_dates, _ = datetime_array_set_to_freq_and_filter(data_dt=df_s.index, ref_datetime=ref_datetime, freq_datetime=freq_datetime, duplicates=dup_datetime, start_datetime=start_datetime, end_datetime=end_datetime)
        if df_s_index_mask_bool.size == 0:
            return (lat_s, lon_s, df_s.iloc[0:0])
        if np.all(~df_s_index_mask_bool):
            return (lat_s, lon_s, df_s.iloc[0:0])
        df_s = df_s.loc[df_s.index[df_s_index_mask_bool]]
        df_s_index_dates = pd.to_datetime(df_s_index_dates)
        df_s.index = df_s_index_dates
    return (lat_s, lon_s, df_s)

def get_df_from_hydroweb_txt(reach_ids: Tuple[str | int, ...], hydroweb_stations: Tuple[str, ...], hydroweb_txt_file_pattern: str | pathlib.PurePath, sword_file_path: str | pathlib.PurePath, x_ref: Literal['node_length', 'dist_out']='node_length', no_data_value: float | str=9999.999, add_facc: bool=False, ref_datetime: datetime=SIC4DVarLowCostDefaults().def_ref_datetime, freq_datetime: str=SIC4DVarLowCostDefaults().def_freq_datetime, dup_datetime: Literal['drop', 'raise']='raise', start_datetime: datetime | float | int | None=None, end_datetime: datetime | float | int | None=None, clean_run: bool=False, debug_mode: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """

    Output is sorted!

    Parameters
    ----------
    reach_ids : Tuple[str | int, ...]
        The reach ids to be loaded in SWORD file to find and assign HydroWeb stations to the nearest nodes.
    hydroweb_stations : Tuple[str, ...]
        The name of the HydroWeb stations
    hydroweb_txt_file_pattern : str | pathlib.PurePath
        the pattern of the file that contains the node water surface elevations.
         It is formatted according to hydroweb_stations.
    sword_file_path : str | pathlib.PurePath | None
        The path to the SWORD netCDF file
    x_ref : Literal["node_length", "dist_out"]
        The reference for the distance.
         Due to inconsistencies with the SWORD, it is recommended to use "node_length".
    no_data_value : float | str
        the no data value in the HydroWeb files.
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
        Whether to print statements while running this function
    debug_mode : bool
        Whether to print debug statements while running this function

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (i) DataFame with the Water Surface Elevation with the nodes as rows and times as columns
        (ii) DataFame with the Uncertainty in Water Surface Elevation with the nodes as rows and times as columns
        (iii) DataFame with the SWORD nodes. Nodes as rows, columns are reach_id, dist (distance from the most upstream
         point) and optionally facc -> accumulated upstream area
    """
    import shapely
    import pyproj
    if clean_run:
        debug_mode = False
    if debug_mode:
        clean_run = False
    reach_ids = [int(i) for i in reach_ids]
    if len(reach_ids) == 0:
        raise TypeError('must specify at lest one reach_id')
    sword_dict = get_vars_from_sword_file(reach_ids=reach_ids, sword_file_path=sword_file_path, node_vars=('reach_id', 'node_id', 'x', 'y', 'node_length', 'dist_out'), reach_vars=('facc',) if add_facc else (), x_ref=x_ref, clean_run=clean_run)
    df_sword = pd.DataFrame(sword_dict['nodes'])[['node_id', 'reach_id', 'x', 'y', 'dist_out']]
    df_sword['node_id'] = df_sword['node_id'].astype(np.int64)
    df_sword['reach_id'] = df_sword['reach_id'].astype(np.int64)
    df_sword['x'] = df_sword['x'].astype(np.float32)
    df_sword['y'] = df_sword['y'].astype(np.float32)
    df_sword['dist_out'] = df_sword['dist_out'].astype(np.float64)
    df_sword['dist'] = df_sword['dist_out'].max() - df_sword['dist_out']
    df_sword.sort_values('dist', inplace=True, ascending=True)
    df_sword.set_index('node_id', inplace=True)
    if add_facc:
        df_sword['facc'] = np.nan
        for r_n, r_id in enumerate(reach_ids):
            df_sword.loc[df_sword['reach_id'] == r_id, 'facc'] = sword_dict['reaches']['facc'][r_n]
    del sword_dict
    utm_proj_s = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), load_utm_crs_from_latlon(latitude=df_sword['y'].mean(), longitude=df_sword['x'].mean()), always_xy=True)
    df_sword['x'], df_sword['y'] = utm_proj_s.transform(df_sword['x'], df_sword['y'])
    msg = f'loading data from {len(hydroweb_stations)} Hydroweb stations'
    if not clean_run:
        print(msg)
    df_wse_all = []
    df_wse_u_all = []
    for hw_s in hydroweb_stations:
        if debug_mode:
            print('loading data for Hydroweb station', hw_s)
        hw_f = str(hydroweb_txt_file_pattern).format(hw_s)
        lat_s, lon_s, df_s = read_hydroweb_txt(hydroweb_txt_file_path=hw_f, no_data_value=no_data_value, ref_datetime=ref_datetime, freq_datetime=freq_datetime, dup_datetime=dup_datetime, start_datetime=start_datetime, end_datetime=end_datetime)
        if debug_mode:
            print(f'data loaded. Station located at ({(lat_s, lon_s)}), (lat, lon)')
        x_s, y_s = utm_proj_s.transform(lon_s, lat_s)
        pt_s = shapely.Point((x_s, y_s))
        if debug_mode:
            print('finding closest node to station')
        dist_0 = np.inf
        node_id_hw = None
        for node_id in df_sword.index:
            pt_i = shapely.Point((df_sword.loc[node_id, 'x'], df_sword.loc[node_id, 'y']))
            dist_i = shapely.distance(pt_s, pt_i)
            if dist_i < dist_0:
                dist_0 = dist_i
                node_id_hw = node_id
        if debug_mode:
            print(f'closest node is: {node_id_hw} located {round(dist_0, 2)}m from Hydroweb station')
        df_wse_hw_i = pd.DataFrame(df_s['wse'].to_numpy(), index=df_s.index, columns=[node_id_hw])
        df_wse_u_hw_i = pd.DataFrame(df_s['wse_u'].to_numpy(), index=df_s.index, columns=[node_id_hw])
        df_wse_all.append(df_wse_hw_i)
        df_wse_u_all.append(df_wse_u_hw_i)
    df_wse_all = pd.concat(df_wse_all, axis=1)
    df_wse_u_all = pd.concat(df_wse_u_all, axis=1)
    df_wse_all.columns = df_wse_all.columns.astype(np.int64)
    df_wse_u_all.columns = df_wse_u_all.columns.astype(np.int64)
    df_wse_all.sort_index(inplace=True)
    df_wse_u_all.sort_index(inplace=True)
    df_wse_all['sec_from_ref'] = (df_wse_all.index - ref_datetime).total_seconds()
    df_wse_all['sec_from_ref'] = df_wse_all['sec_from_ref'].astype(np.int64)
    df_wse_all.set_index('sec_from_ref', inplace=True)
    df_wse_u_all.index = df_wse_all.index
    node_ids = df_wse_all.columns.to_numpy(dtype=np.int64)
    node_dist = df_sword.loc[node_ids, 'dist'].to_numpy(dtype=np.float64)
    node_ids = node_ids[np.argsort(node_dist)]
    df_wse_all = df_wse_all[node_ids]
    df_wse_u_all = df_wse_u_all[node_ids]
    df_wse_all = df_wse_all.T
    df_wse_u_all = df_wse_u_all.T
    df_wse_all = df_wse_all.astype(np.float32)
    df_wse_u_all = df_wse_u_all.astype(np.float32)
    if add_facc:
        df_sword = df_sword[['reach_id', 'dist', 'facc']]
    else:
        df_sword = df_sword[['reach_id', 'dist']]
    if not clean_run:
        print(msg, 'done')
    return (df_wse_all, df_wse_u_all, df_sword)
if __name__ == '__main__':
    '  '
    base_path = pathlib.Path('C:\\Users\\isadora.rezende\\PhD\\Datasets')
    df_wse_all_test, df_wse_u_all_test, df_sword_test = get_df_from_hydroweb_txt(hydroweb_stations=('FIUME-OGLIO_KM0191', 'PO_KM0160', 'PO_KM0145', 'PO_KM0140', 'PO_KM0098', 'PO_KM0096', 'PO_KM0082'), hydroweb_txt_file_pattern=base_path / 'Hydroweb' / 'Po' / 'download_expert' / 'R_PO_{}_exp.txt', sword_file_path=base_path / 'SWORD' / 'v15' / 'netcdf' / 'eu_sword_v15.nc', reach_ids=(21406100011, 21406100021, 21406100031, 21406100041, 21406100051, 21406100061, 21406100071, 21406100081, 21406100101, 21406100111), add_facc=True, freq_datetime='3h', start_datetime=datetime(2008, 5, 1), end_datetime=datetime(2009, 6, 1))
    print(df_wse_all_test)
    print(df_sword_test)