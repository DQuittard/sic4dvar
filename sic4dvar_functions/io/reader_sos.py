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

Last modified on May 20th 2024 at 22:00
by @Isadora Silva

@authors: Isadora Silva
"""
import copy
import pathlib
from typing import Tuple
import netCDF4 as nc4
import numpy as np
from sic4dvar_functions.helpers.helpers_arrays import masked_array_to_nan_array, array_as_col_vector

def get_q_prior_from_sos_file(reach_ids: Tuple[str | int, ...], sos_file_path: str | pathlib.PurePath, clean_run: bool=False) -> dict:
    """
    Read the discharge information from SWORD of Science netcdf file and return a dictionary with the unmasked arrays.

    Parameters
    ----------
    reach_ids : Tuple[str, ...]
        The reach ids to be read from the SWORD file
    sos_file_path : str | pathlib.PurePath
        The path to the SWORD of Science netCDF file
    clean_run
        Whether to print statements while running this function

    Returns
    -------
    dict
        Dictionary with the unmasked arrays. Keys are "min_q", "mean_q" and "max_q".
    """
    msg = 'loading discharge data from SWORD of Science'
    if not clean_run:
        print(msg)
    sos_file_path = pathlib.Path(sos_file_path)
    if not sos_file_path.exists():
        raise FileNotFoundError(f'sos_file_path {sos_file_path} does not exist')
    q_prior_dict = {'min_q': None, 'mean_q': None, 'max_q': None}
    reach_ids = [int(i) for i in reach_ids]
    with nc4.Dataset(sos_file_path) as sos_ds:
        sos_reach_ids = sos_ds['reaches']['reach_id'][:]
        sos_index_array = np.isin(sos_reach_ids, reach_ids)
        if np.any(sos_index_array):
            sos_index_array = sos_index_array.nonzero()[0]
        else:
            sos_index_array = np.empty(0, dtype=np.int32)
        if sos_index_array.size != len(reach_ids):
            raise IndexError(f'Sword of Science file does not contain data for some reaches')
        for q in q_prior_dict.keys():
            q_prior_array = copy.deepcopy(sos_ds['model'][q][sos_index_array])
            q_prior_array = masked_array_to_nan_array(q_prior_array)
            q_prior_dict[q] = array_as_col_vector(q_prior_array)
    if np.all(np.isnan(q_prior_dict['mean_q'])):
        raise RuntimeError('all q_mean priors are NaN')
    if not clean_run:
        print(msg, 'done')
    return q_prior_dict
if __name__ == '__main__':
    base_path = pathlib.Path('C:\\Users\\isadora.rezende\\PhD')
    print(get_q_prior_from_sos_file(reach_ids=(21406100031, 21406100041, 21406100051, 21406100061, 21406100081, 21406100101, 21406100111), sos_file_path=base_path / 'Datasets' / 'SOS' / 'v15' / 'constrained' / 'eu_sword_v15_SOS_priors.nc'))