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
import numpy as np
import pandas as pd

def reorder_ids_with_indices(ids, sword_node_order=None, params=None):
    indexed_ids = list(enumerate(ids))
    if sword_node_order is not None:
        indexed_ids = list(zip(indexed_ids, sword_node_order))
        indexed_ids.sort(key=lambda x: x[1])
        sorted_ids = [x[0] for x in indexed_ids]
    else:
        sorted_ids = sorted(indexed_ids, key=lambda x: x[1])
    if not params.start_from_downstream:
        sorted_ids.reverse()
    indexes = [index for index, _ in sorted_ids]
    values = [id_value for _, id_value in sorted_ids]
    return (indexes, values)

def check_na(value):
    """ check if the specified value is None, '', pd.na, np.nan, is_empty or masked """
    if value is None:
        return True
    if value == '':
        return True
    if value == '--':
        return True
    if value is np.ma.masked:
        return True
    if np.ma.is_masked(value):
        return True
    if isinstance(value, np.ma.core.MaskedConstant):
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    try:
        if np.isnan(value):
            return True
    except TypeError:
        pass
    try:
        if value.mask:
            return True
    except AttributeError:
        pass
    try:
        if value.is_empty:
            return True
    except AttributeError:
        pass
    try:
        if value < -100000000.0:
            return True
    except TypeError:
        pass
    return False

def verify_name_length(name, extension=''):
    if len(name) > 200:
        new_name = name[0:200]
        new_name = new_name + '[...]' + extension
    else:
        new_name = name
    return new_name