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

Created on September 11th 2023 at 16:00
by @Isadora Silva

Last modified on May 15th 2024 at 14:00
by @Isadora Silva

@authors: Isadora Silva
"""
import copy
import itertools
import logging
import math
import operator
from datetime import datetime
from typing import Iterable, Tuple, Any, Literal
import numpy as np
import pandas as pd
from sic4dvar_classes.sic4dvar_0_defaults import SIC4DVarLowCostDefaults
from sic4dvar_functions.helpers.helpers_generic import pairwise, dt_as_pd_datetime, PdTimeDeltaFreq

def iterable_to_flattened_array(my_iterable: Iterable) -> np.ndarray:
    """
    Convert to array/masked array and flatten.
    Creates a copy
    """
    my_array1 = copy.deepcopy(my_iterable)
    if not isinstance(my_array1, np.ndarray) or not isinstance(my_array1, np.ma.MaskedArray):
        my_array1 = np.array(my_array1)
    else:
        my_array1 = my_array1
    if my_array1.ndim != 1:
        my_array1 = my_array1.flatten()
    return my_array1

def masked_array_to_nan_array(my_array: np.ma.MaskedArray):
    """
    If array is a masked array, set masked values in array to np.nan.
    Might change the data type to float32 / float64
    Creates a copy.
    """
    my_array1 = copy.deepcopy(my_array)
    if my_array1.dtype == np.int32 or my_array1.dtype == np.int64:
        if my_array1.size == 0:
            my_array1 = np.empty(0, dtype=np.float32)
            return my_array1
        if isinstance(my_array1, np.ma.MaskedArray):
            max_value = np.ma.max(my_array1)
        else:
            max_value = np.nanmax(my_array1)
        if my_array1.dtype == np.int32:
            if not np.isnan(max_value):
                error_msg = 'max value of array is bigger than max allowed valued in float 32'
                if max_value > np.finfo(np.float32).max:
                    raise RuntimeError(error_msg)
            my_array1 = my_array1.astype(np.float32)
            return my_array1
        if my_array1.dtype == np.int64:
            if not np.isnan(max_value):
                error_msg = 'max value of array is bigger than max allowed valued in float 64'
                if max_value > np.finfo(np.float64).max:
                    raise RuntimeError(error_msg)
            my_array1 = my_array1.astype(np.float64)
            return my_array1
    if isinstance(my_array1, np.ma.MaskedArray):
        my_array1 = my_array1.filled(np.nan)
    return my_array1

def nan_array_to_masked_array(my_array: np.ndarray | np.ma.MaskedArray, fill_value: None | float | int=None) -> np.ma.MaskedArray:
    """
    If array is not masked array, set NaN values to masked.
    Option to set the fill value of the masked array.
    Creates a copy.
    """
    my_array1 = copy.deepcopy(my_array)
    if not isinstance(my_array1, np.ma.MaskedArray):
        my_array1 = np.ma.masked_invalid(my_array1)
    if fill_value is not None:
        my_array1.set_fill_value(fill_value)
    return my_array1

def arrays_rmv_nan_pair(x0: Iterable, y0: Iterable) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove NaN data considering both arrays.
    Data will be flattened (1 dimension!)
    Both arrays must be the same size.
    E.g.: x = [1, 2, np.nan, 3], y = [4, np.nan, 5, 6] -> x = [1, 3], y = [4, 6]

    Parameters
    ----------
    x0 : Iterable
        The x values
    y0: Iterable
        The y values

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        NaN free (i) x, (ii) y, and (iii) flattened index arrays
    """
    x, y = (iterable_to_flattened_array(x0), iterable_to_flattened_array(y0))
    x = masked_array_to_nan_array(x)
    y = masked_array_to_nan_array(y)
    if x.size != y.size:
        raise TypeError('for removing NaN in pairs, x and y must be the same size')
    df = pd.DataFrame({'x': x, 'y': y})
    df.dropna(axis=0, how='any', inplace=True)
    x, y, i = (np.array(df['x']), np.array(df['y']), np.array(df.index, dtype=np.int32))
    return (x, y, i)

def arrays_rmv_next_same(x0: np.ndarray | Iterable, y0: np.ndarray | Iterable, float_atol: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove a pair of entries that are the same in a sequence.

    Parameters
    ----------
    x0 : np.ndarray | Iterable
        The x values
    y0: np.ndarray | Iterable
        The y values
    float_atol : float
        floating-point representation error tolerance, used when comparing floats with function math.isclose() to
        remove duplicates. The greater this number, the more numbers will be considered the same (the shorter the x and
         y arrays will be).
        E.g.: 0.1 + 0.2 == 0.3 -> False. math.isclose(0.1 + 0.2, 0.3, abs_tol=float_atol) -> True.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Duplicate in a sequence free (i)x, (ii)y and (iii)indexes arrays
    """
    x, y = [iterable_to_flattened_array(i_) for i_ in [x0, y0]]
    if x.size != y.size:
        raise TypeError('for removing consecutive entries that are the same, x and y must be the same size')
    x_is_int = True if np.issubdtype(x.dtype, np.integer) else False
    y_is_int = True if np.issubdtype(y.dtype, np.integer) else False
    no_dup_x_list = [x[0]]
    no_dup_y_list = [y[0]]
    no_dup_n_list = [0]
    n = 0
    for x_i, y_i in zip(x[1:], y[1:]):
        n += 1
        if x_is_int:
            x_comp = x_i == no_dup_x_list[-1]
        else:
            x_comp = math.isclose(x_i, no_dup_x_list[-1], abs_tol=float_atol)
        if x_comp:
            if y_is_int:
                y_comp = y_i == no_dup_y_list[-1]
            else:
                y_comp = math.isclose(y_i, no_dup_y_list[-1], abs_tol=float_atol)
            if y_comp:
                continue
        no_dup_x_list.append(x_i)
        no_dup_y_list.append(y_i)
        no_dup_n_list.append(n)
    x = np.array(no_dup_x_list)
    y = np.array(no_dup_y_list)
    n = np.array(no_dup_n_list, dtype=np.int32)
    return (x, y, n)

def sort_by_distance_to_value(my_array: np.ndarray | Iterable, value: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort the array according to the absolute distance to the value

    Parameters
    ----------
    my_array : np.ndarray
        The array to be sorted
    value : Any
        The value that will be used as a base

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the arrays of the indexes[0] and values[1]
    """
    if not isinstance(my_array, np.ndarray):
        my_array = np.array(my_array)
    my_array_1 = copy.deepcopy(my_array)
    difs = np.abs(my_array_1 - value)
    ids = np.argsort(difs)
    my_array_1 = my_array_1[ids]
    return (ids, my_array_1)

def find_nearest(my_array: np.ndarray | Iterable, value) -> Tuple[int, float]:
    """
    Find the nearest value in a numpy array

    Parameters
    ----------
    my_array : np.ndarray
        The array in which the nearest entry to value will be selected
    value : Any
        The value that will be searched in the array

    Returns
    -------
    Tuple[int, Any]
        A tuple containing the nearest index to value and the value itself
    """
    if not isinstance(my_array, np.ndarray):
        my_array = np.array(my_array)
    i = np.abs(my_array - value).argmin()
    v = my_array[i]
    return (i, v)

def find_n_nearest(my_array: np.ndarray | Iterable, value: Any, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the n nearest values in a numpy array and their indexes

    Parameters
    ----------
    my_array : np.ndarray
        The array in which the n nearest entries to value will be selected
    value : Any
        The value that will be searched in the array
    n : int
        The n nearest values

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the arrays of the n nearest indexes[0] and values[1]
    """
    if n < 1:
        raise TypeError('n must be bigger than 1')
    ids, my_array = sort_by_distance_to_value(my_array, value)
    return (ids[:n], my_array[:n])

def find_normalized_nearest(my_array0: np.ndarray, value: float, norm_diff_thr: float=0.1, min_n: int=1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select the n values of my_array0 that are nearest (in absolute terms) to the desired value.
    n is defined as the maximum between min_n and the number of values whose normalized difference is smaller
     than norm_diff_thr.

    Output is not sorted.

    Parameters
    ----------
    my_array0 : nd.array
        The array in which the nearest entries to value will be selected
    value : float
        The value that will be searched in the array
    norm_diff_thr: float
        The normalized difference threshold to define the number of entries that will be selected.
    min_n : int
        The minimum number of points to be returned

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (i) The indices of the flattened array that are closer (in normalized absolute terms) to the target value.
        (ii) The normalized weights [0, 1] of the indices given in (i), the closer the value the higher the weight.
    """
    my_array = copy.deepcopy(my_array0).flatten()
    n_valid_pts = np.count_nonzero(np.isfinite(my_array))
    req_n_pts = min(max(1, min_n), n_valid_pts)
    diff_array = np.full_like(my_array, fill_value=np.nan)
    diff_array[np.isfinite(my_array)] = np.abs(my_array[np.isfinite(my_array)] - value)
    min_, max_ = (np.nanmin(diff_array), np.nanmax(diff_array))
    if min_ == max_:
        diff_array[np.isfinite(diff_array)] = 0.0
    else:
        diff_array = (diff_array - min_) / (max_ - min_)
    idxs_array = np.nonzero(diff_array.flatten() <= norm_diff_thr)[0]
    if idxs_array.size < req_n_pts:
        idxs_array = np.argsort(diff_array.flatten())[:req_n_pts]
    return (idxs_array, 1 - diff_array.flatten()[idxs_array])

def find_relative_nearest(my_array0: np.ndarray, value: float, rel_diff_thr: float=0.1, min_n: int=1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select the n values of my_array0 that are nearest (in absolute terms) to the desired value.
    n is defined as the maximum between min_n and the number of values whose relative difference is smaller
     than rel_diff_thr.

    Output is not sorted.

    Parameters
    ----------
    my_array0 : nd.array
        The array in which the nearest entries to value will be selected
    value : float
        The value that will be searched in the array
    rel_diff_thr: float
        The relative difference threshold to define the number of entries that will be selected.
    min_n : int
        The minimum number of points to be returned

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (i) The indices of the flattened array that are closer (in normalized absolute terms) to the target value.
        (ii) The relative weights [0, 1] of the indices given in (i), the closer the value the higher the weight.
    """
    my_array = copy.deepcopy(my_array0).flatten()
    n_valid_pts = np.count_nonzero(np.isfinite(my_array))
    req_n_pts = min(max(1, min_n), n_valid_pts)
    diff_array = np.full_like(my_array, fill_value=np.nan)
    diff_array[np.isfinite(my_array)] = np.abs(my_array[np.isfinite(my_array)] - value)
    diff_array /= value
    idxs_array = np.nonzero(diff_array.flatten() <= rel_diff_thr)[0]
    if idxs_array.size < req_n_pts:
        idxs_array = np.argsort(diff_array.flatten())[:req_n_pts]
    return (idxs_array, np.exp(-diff_array[idxs_array].flatten()))

def _arrays_de_in_crease(my_array: np.ndarray | Iterable, comp_op: operator, check: bool, force: bool, remove_nan: bool=True) -> np.ndarray | bool:
    if check and force:
        raise TypeError('only one of force or check can be True')
    if not check and (not force):
        raise TypeError('one of force or check must be True')
    if not isinstance(my_array, np.ndarray):
        my_array = np.array(my_array)
    if my_array.ndim != 1:
        raise TypeError('can only check/force increase/decrease with 1D arrays')
    my_array1 = copy.deepcopy(my_array)
    if remove_nan:
        my_array1 = my_array1[np.isfinite(my_array1)]
    for n_0i, n_1i in pairwise(range(my_array1.shape[0])):
        v_0i, v_1i = (my_array1[n_0i], my_array1[n_1i])
        if comp_op(v_1i, v_0i):
            if check:
                return False
            my_array1[n_1i] = v_0i
    if check:
        return True
    return my_array1

def arrays_check_decrease(my_array: np.ndarray | Iterable, remove_nan: bool=True) -> bool:
    return _arrays_de_in_crease(my_array, comp_op=operator.gt, check=True, force=False, remove_nan=remove_nan)

def arrays_check_increase(my_array: np.ndarray | Iterable, remove_nan: bool=True) -> bool:
    return _arrays_de_in_crease(my_array, comp_op=operator.lt, check=True, force=False, remove_nan=remove_nan)

def arrays_force_decrease(my_array: np.ndarray | Iterable, remove_nan: bool=True) -> np.ndarray:
    return _arrays_de_in_crease(my_array, comp_op=operator.gt, check=False, force=True, remove_nan=remove_nan)

def arrays_force_increase(my_array: np.ndarray | Iterable, remove_nan: bool=True) -> np.ndarray:
    return _arrays_de_in_crease(my_array, comp_op=operator.lt, check=False, force=True, remove_nan=remove_nan)

def get_mask_nan_across_arrays(*my_arrays: np.ndarray | np.ma.MaskedArray) -> np.ndarray:
    expected_shape = None
    sum_array = np.empty(0)
    if not isinstance(my_arrays, tuple):
        my_arrays = (my_arrays,)
    for arr in my_arrays:
        if arr.ndim > 2:
            raise TypeError('only works with 1D or 2D arrays')
        arr1 = copy.deepcopy(arr)
        arr1 = masked_array_to_nan_array(arr1)
        if expected_shape is None:
            expected_shape = arr.shape
            sum_array = copy.deepcopy(arr1)
            continue
        if arr.shape != expected_shape:
            raise TypeError('shape of all arrays should be the same')
        sum_array += arr1
        if np.any(np.isfinite(sum_array)):
            sum_array /= np.nanmin(sum_array)
    nan_bool_array = np.isnan(sum_array)
    return nan_bool_array

def get_index_valid_data(*my_arrays: np.ndarray | np.ma.MaskedArray, drop_na_how: Literal['any', 'all']='any', axis: int | tuple=(0, 1)) -> tuple:
    """
    Parameters
    ----------
    my_arrays : np.ndarray | np.ma.MaskedArray
    drop_na_how : Literal["any", "all"]
        used in in 2D arrays. Pandas dropna argument: determine if row or column is removed, when we have at least one
         NaN or all NaN.
    axis: tuple
        axis (in order) to use pandas dropna in 2D arrays

    Returns
    -------
    tuple
        tuple with the valid indexes.
    """
    if isinstance(axis, int):
        axis = (axis,)
    if not isinstance(my_arrays, tuple):
        my_arrays = (my_arrays,)
    nan_bool_array = get_mask_nan_across_arrays(*my_arrays)
    expected_shape = nan_bool_array.shape
    tmp_array = np.full_like(nan_bool_array, fill_value=1.0, dtype=np.float32)
    tmp_array[nan_bool_array] = np.nan
    if not np.any(nan_bool_array):
        idx_dim0 = np.array(range(expected_shape[0]))
        if len(expected_shape) == 1:
            return (idx_dim0,)
        return (idx_dim0, np.array(range(expected_shape[1])))
    if np.all(nan_bool_array):
        return (np.array([], dtype=np.int32),) * len(expected_shape)
    if len(expected_shape) == 1:
        return ((~nan_bool_array).nonzero()[0],)
    df_valid = pd.DataFrame(copy.deepcopy(tmp_array))
    for ax in axis:
        df_valid = df_valid.dropna(axis=ax, how=drop_na_how)
    if df_valid.empty:
        return (np.array([], dtype=np.int32),) * len(expected_shape)
    return (np.array(df_valid.index, dtype=np.int32), np.array(df_valid.columns, dtype=np.int32))

def array_as_row_vector(my_array: np.ndarray | np.ma.MaskedArray):
    my_vector = copy.deepcopy(my_array)
    try:
        my_vector.shape = (1, my_array.size)
    except AttributeError as ae:
        if "'list' object has no attribute 'size'" in ae.args[0]:
            my_vector = np.array(my_vector)
            my_vector.shape = (my_vector.size, 1)
        else:
            raise ae
    return my_vector

def array_as_col_vector(my_array: np.ndarray | np.ma.MaskedArray):
    my_vector = copy.deepcopy(my_array)
    try:
        my_vector.shape = (my_array.size, 1)
    except AttributeError as ae:
        if "'list' object has no attribute 'size'" in ae.args[0]:
            my_vector = np.array(my_vector)
            my_vector.shape = (my_vector.size, 1)
        else:
            raise ae
    return my_vector

def check_shape(my_array: np.ndarray | np.ma.MaskedArray, expected_shape: tuple, force_shape: bool=False):
    if not isinstance(my_array, np.ndarray) or not isinstance(my_array, np.ma.MaskedArray):
        my_array = np.array(my_array)
    init_shape = my_array.shape
    if isinstance(expected_shape, int):
        expected_shape = (expected_shape,)
    error_msg = f'array must have shape {expected_shape}, not {init_shape}'
    if init_shape == expected_shape:
        return copy.deepcopy(my_array)
    if not force_shape:
        raise IndexError(error_msg)
    if len(expected_shape) == 1:
        is_vector = False
        is_1d = True
    elif len(expected_shape) == 2:
        is_vector = np.any([i_ == 1 for i_ in expected_shape])
        is_1d = False
    else:
        raise IndexError(error_msg)
    if not (is_1d or is_vector):
        print('debug helper')
        print(my_array.shape)
        raise IndexError(error_msg)
    if is_1d:
        major_size = expected_shape[0]
    else:
        try:
            major_size = [i_ for i_ in expected_shape if i_ != 1][0]
        except IndexError:
            major_size = 1
    tmp_match_dim = [i_ == major_size for i_ in init_shape]
    if not np.any(tmp_match_dim):
        raise IndexError(error_msg)
    if len(tmp_match_dim) == 1:
        my_array_out = copy.deepcopy(my_array)
        if is_1d:
            raise AssertionError('no matching indexes, but this condition should have already happened')
    elif np.all(tmp_match_dim):
        raise IndexError(error_msg)
    elif tmp_match_dim[0]:
        my_array_out = copy.deepcopy(my_array[:, 0])
    else:
        my_array_out = copy.deepcopy(my_array[0, :])
    if is_vector:
        if expected_shape[0] == 1:
            my_array_out = array_as_row_vector(my_array_out)
        elif expected_shape[1] == 1:
            my_array_out = array_as_col_vector(my_array_out)
        else:
            raise AssertionError('no dimension with size 1, but this condition should have already happened')
    if my_array_out.shape != expected_shape:
        raise AssertionError(f'opposite logic here somewhere, expected {expected_shape} got {my_array_out.shape}, invert code')
    return my_array_out

def array_filter_by_index(my_array: np.ndarray, row_0_ids: np.ndarray | None=None, col_0_ids: np.ndarray | None=None):
    """ filter the array by the indexes specified in row and col, keep shape of array (use for row/col vectors!) """
    if my_array.size == 0:
        return copy.deepcopy(my_array)
    if my_array.ndim > 2:
        raise TypeError('can only index in 1D or 2D arrays')
    row_ids = copy.deepcopy(row_0_ids)
    col_ids = copy.deepcopy(col_0_ids)
    if row_ids is None and col_ids is None:
        raise TypeError('one of row_ids or col_ids must be specified')
    if row_ids is not None:
        row_ids = row_ids.astype(np.int32)
        row_ids = row_ids.flatten()
    if col_ids is not None:
        col_ids = col_ids.astype(np.int32)
        col_ids = col_ids.flatten()
    if my_array.ndim == 1:
        if row_ids is not None and col_ids is not None:
            raise TypeError('only one of row_ids or col_ids can be specified when array is 1D')
        for id_array in (row_ids, col_ids):
            if id_array is not None:
                if id_array.ndim != 1:
                    if np.any([i_ == 1 for i_ in id_array.shape]):
                        id_array = id_array.flatten()
                    else:
                        raise IndexError(f'cannot use a {id_array.ndim}D array to index a 1D array')
                return copy.deepcopy(my_array[id_array])
    if row_ids is None or np.array_equal(row_ids, np.array(range(my_array.shape[0]))):
        out_array = copy.deepcopy(my_array)
        row_ids = np.array(range(my_array.shape[0]))
    else:
        try:
            out_array = copy.deepcopy(my_array[row_ids, :])
        except IndexError as ie:
            logging.debug('debugger helper')
            logging.debug(f'array: {my_array.shape}')
            logging.debug(f'{my_array}')
            logging.debug(f'row ids: {row_ids.shape}')
            logging.debug(f'{row_ids}')
            raise ie
    if col_ids is None or np.array_equal(col_ids, np.array(range(my_array.shape[1]))):
        col_ids = np.array(range(my_array.shape[1]))
    else:
        try:
            out_array = out_array[:, col_ids]
        except IndexError as ie:
            logging.debug('debugger helper')
            logging.debug(f'array: {out_array.shape}')
            logging.debug(f'{out_array}')
            logging.debug(f'col ids: {col_ids.shape}')
            logging.debug(f'{col_ids}')
            raise ie
    return check_shape(out_array, (row_ids.size, col_ids.size), force_shape=True)

def array_fill_with_nan_from_ids(my_array: np.ndarray, valid_0_ids: np.ndarray, expected_shape: tuple | None):
    if not isinstance(valid_0_ids, np.ndarray):
        valid_0_ids = np.array(valid_0_ids)
    if not isinstance(my_array, np.ndarray):
        my_array = np.array(my_array)
    if my_array.ndim != valid_0_ids.ndim:
        raise TypeError(f'number of dimensions do not match {my_array.ndim} != {valid_0_ids.ndim}')
    if valid_0_ids.size > 0:
        if valid_0_ids.ndim == 1:
            if expected_shape is None:
                expected_shape = (np.max(valid_0_ids) + 1,)
        elif expected_shape is None:
            expected_shape = (np.max(valid_0_ids[:, 0]) + 1, np.max(valid_0_ids[:, 1]) + 1)
    elif expected_shape is None:
        raise TypeError('must specify expected shape if valid_0_ids is empty')
    expected_size = 1
    for i in expected_shape:
        expected_size *= i
    if valid_0_ids.size > 0:
        if valid_0_ids.ndim == 1:
            valid_flat_ids = copy.deepcopy(valid_0_ids)
        else:
            valid_flat_ids = []
            for pair_ids in valid_0_ids:
                try:
                    valid_flat_ids.append(np.ravel_multi_index(pair_ids, dims=expected_shape))
                except ValueError as ve:
                    print('debug helper')
                    print('expected_shape', expected_shape)
                    print('pair ids', pair_ids)
                    raise ve
            valid_flat_ids = np.array(valid_flat_ids)
    else:
        valid_flat_ids = np.empty(0, dtype=np.int32)
    out_array = np.full(expected_size, fill_value=np.nan, dtype=np.float32)
    out_array[valid_flat_ids] = my_array.flatten()
    out_array.shape = expected_shape
    return check_shape(out_array, expected_shape, force_shape=True)

def array_fix_next_same(my_array: np.ndarray, float_atol: float=SIC4DVarLowCostDefaults().def_float_atol) -> np.ndarray:
    """ loop in pair of points in array and add float_atol to the second value if they are the same """
    out_array = copy.deepcopy(my_array)
    is_next_same = True
    while is_next_same:
        is_next_same = False
        for (n0, v0), (n1, v1) in pairwise(enumerate(out_array)):
            if np.isclose(v0, v1, rtol=0.0, atol=float_atol):
                out_array[n1] = v1 + float_atol
                is_next_same = True
    return out_array

def arrays_bounds(ref_array: np.ndarray, value0_low_bound_array: np.ndarray | np.ma.MaskedArray | None=None, value0_up_bound_array: np.ndarray | np.ma.MaskedArray | None=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (1) value_0_low_bound_array
        (2) value_0_up_bound_array
    """
    def_low_array, def_up_array = [np.full_like(ref_array, fill_value=inf_, dtype=ref_array.dtype) for inf_ in [-np.inf, np.inf]]
    if value0_low_bound_array is None:
        value_0_low_bound_array = def_low_array
    else:
        value_0_low_bound_array = check_shape(masked_array_to_nan_array(value0_low_bound_array), expected_shape=ref_array.shape, force_shape=False)
        value_0_low_bound_array[np.isnan(value_0_low_bound_array)] = -np.inf
    if value0_up_bound_array is None:
        value_0_up_bound_array = def_up_array
    else:
        value_0_up_bound_array = check_shape(masked_array_to_nan_array(value0_up_bound_array), expected_shape=ref_array.shape, force_shape=False)
        value_0_up_bound_array[np.isnan(value_0_up_bound_array)] = np.inf
    return (value_0_low_bound_array, value_0_up_bound_array)

class DataDateTimeArrays(PdTimeDeltaFreq):

    def __init__(self, data_dt: np.ndarray | pd.Series | tuple | list, ref_datetime: datetime, freq_datetime: str):
        """ attributes are not sorted """
        super().__init__(freq_datetime=freq_datetime)
        data_dt = np.array(data_dt).flatten()
        if not isinstance(data_dt[0], np.datetime64):
            data_dt = pd.to_timedelta(data_dt, unit='second') + ref_datetime
        self._data_dt_timerange: pd.DatetimeIndex = pd.to_datetime(data_dt).round(self._freq_dt_str)
        self._data_dt_sec_array = np.array((self._data_dt_timerange - ref_datetime).total_seconds(), dtype=np.int64)

    @property
    def data_dt_timerange(self) -> pd.DatetimeIndex:
        return copy.deepcopy(self._data_dt_timerange)

    @property
    def data_dt_sec_array(self) -> np.ndarray:
        return copy.deepcopy(self._data_dt_sec_array)

def datetime_create_time_range(start_datetime: datetime | float | int, end_datetime: datetime | float | int, ref_datetime: datetime, freq_datetime: str):
    freq_o = PdTimeDeltaFreq(freq_datetime=freq_datetime)
    freq_dt_int, freq_dt_str, freq_datetime = (freq_o.freq_datetime_int, freq_o.freq_datetime_str, freq_o.freq_datetime)
    start_datetime = dt_as_pd_datetime(dt=start_datetime, ref_datetime=ref_datetime, pd_freq=freq_dt_str)
    end_datetime = dt_as_pd_datetime(dt=end_datetime, ref_datetime=ref_datetime, pd_freq=freq_dt_str)
    stat_dt = start_datetime.normalize()
    end_dt = end_datetime.normalize() + (pd.to_timedelta(1, unit='D') + pd.to_timedelta(freq_dt_int, unit=freq_dt_str))
    all_dt_timerange = pd.date_range(start=stat_dt, end=end_dt, freq=freq_datetime, inclusive='both')
    all_dt_timerange = all_dt_timerange[all_dt_timerange <= end_datetime]
    return all_dt_timerange

def datetime_array_set_to_freq_and_filter(data_dt: np.ndarray | pd.Series | tuple | list, ref_datetime: datetime, freq_datetime: str, duplicates: Literal['drop', 'keep', 'raise']='raise', start_datetime: datetime | float | int | None=None, end_datetime: datetime | float | int | None=None):
    """
    Match the datetimes to the specified frequency
    Filter the datetime array to include only the values between start and end dates.

    Parameters
    ----------
    data_dt : np.ndarray | pd.Series | tuple | list
        the datetime of the data, either as datetime objects or as seconds from reference (int and float)
    ref_datetime : datetime
        The reference datetime to convert between dates to seconds from reference.
    freq_datetime : datetime
        The frequency to match the observations. multiplier and str "D", "h", "min", "s". E.g.: "30min", "3h", "D".
    duplicates: Literal["drop", "keep", "raise"]
        What to do when the data is assigned to the same date. Either "drop", "keep" or "raise": "drop" drops all
         occurrences except the first one; "keep" keeps all the occurrences, "raise" raises IndexError.
    start_datetime : datetime | float | int | None
        start datetime for filtering the data (>=). Either as datetime or as seconds from reference (float or int).
    end_datetime : datetime | float | int | None
        end datetime for filtering the data (<=). Either as datetime or as seconds from reference (float or int).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (i) Boolean array to mask the data according to start and end dates
        (ii) Array with the dates as datetime objects
        (iii) Array with the dates as seconds from reference
    """
    dt_o = DataDateTimeArrays(data_dt=data_dt, ref_datetime=ref_datetime, freq_datetime=freq_datetime)
    freq_dt_int, freq_dt_str, freq_datetime = (dt_o.freq_datetime_int, dt_o.freq_datetime_str, dt_o.freq_datetime)
    data_dt_timerange = dt_o.data_dt_timerange
    data_dt_sec_array = dt_o.data_dt_sec_array
    del data_dt, dt_o
    start_datetime = dt_as_pd_datetime(dt=start_datetime, ref_datetime=ref_datetime, pd_freq=freq_dt_str)
    end_datetime = dt_as_pd_datetime(dt=end_datetime, ref_datetime=ref_datetime, pd_freq=freq_dt_str)
    mask_bool = np.full(len(data_dt_timerange), fill_value=True, dtype=bool)
    if start_datetime is not None:
        mask_bool = data_dt_timerange >= start_datetime
    if end_datetime is not None:
        mask_bool = mask_bool & (data_dt_timerange <= end_datetime)
    data_dt_timerange = data_dt_timerange[mask_bool]
    data_dt_sec_array = data_dt_sec_array[mask_bool]
    if data_dt_timerange.size == 0:
        return (mask_bool, np.empty(0, dtype=np.datetime64), np.empty(0, dtype=np.int64))
    mask_id0 = np.argmax(mask_bool)
    if freq_datetime.startswith('1s'):
        return (mask_bool, data_dt_timerange, data_dt_sec_array)
    all_dt_timerange = datetime_create_time_range(start_datetime=data_dt_timerange[0], end_datetime=data_dt_timerange[-1], ref_datetime=ref_datetime, freq_datetime=freq_datetime)
    all_dt_sec_array = np.array((all_dt_timerange - ref_datetime).total_seconds(), dtype=np.int64)
    bisect_low_id, keep_ids = (0, [])
    for n_sec, t_sec in enumerate(data_dt_sec_array):
        all_id = np.argmin(np.abs(all_dt_sec_array[bisect_low_id:] - t_sec)) + bisect_low_id
        if n_sec > 0 and all_id == bisect_low_id:
            if duplicates == 'keep':
                pass
            elif duplicates == 'drop':
                mask_bool[mask_id0 + n_sec] = False
                continue
            else:
                e_loc = f'{all_dt_timerange[all_id]} for {all_dt_timerange[n_sec - 1]} and {all_dt_timerange[n_sec]}'
                raise IndexError(f'duplicated data considering frequency {freq_datetime}: {e_loc}')
        keep_ids.append(all_id)
        bisect_low_id = all_id
    n_val_elems = np.count_nonzero(mask_bool)
    n_kept_elems = len(keep_ids)
    if n_val_elems != n_kept_elems:
        raise AssertionError(f'number of valid elements {n_val_elems} does not match kept elements {n_kept_elems}')
    all_dt_timerange = all_dt_timerange[keep_ids]
    all_dt_sec_array = all_dt_sec_array[keep_ids]
    return (mask_bool, all_dt_timerange, all_dt_sec_array)
if __name__ == '__main__':
    test_array = np.array(range(4))
    print('1D array', test_array)
    print('as row vector:', array_as_row_vector(test_array))
    print('as column vector:', array_as_col_vector(test_array))
    test_val_array = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    test_val_2d_ids = np.array([[r_n, t_n] for r_n, t_n in itertools.product([0, 1, 2], [6, 7, 8])])
    print(array_fill_with_nan_from_ids(test_val_array, valid_0_ids=test_val_2d_ids, expected_shape=None))
    test_val_2d_ids = np.array([[r_n, t_n] for r_n, t_n in itertools.product([5, 6, 7], [1, 2, 3])])
    print(array_fill_with_nan_from_ids(test_val_array, valid_0_ids=test_val_2d_ids, expected_shape=None))
    test_val_2d_ids = np.array([[r_n, t_n] for r_n, t_n in itertools.product([1, 3, 6], [0, 2, 5])])
    print(array_fill_with_nan_from_ids(test_val_array, valid_0_ids=test_val_2d_ids, expected_shape=None))