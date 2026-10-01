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

Created in September 2023
by @Isadora Silva

Last modified on March 24th 2024 at 18:00
by @Isadora Silva

@authors: Isadora Silva
"""
import itertools
import math
import os
import pathlib
import re
from datetime import datetime
from typing import Tuple
import numpy as np
import pandas as pd
from sic4dvar_classes.sic4dvar_0_defaults import SIC4DVarLowCostDefaults
sic_def = SIC4DVarLowCostDefaults()
try:
    pairwise = itertools.pairwise
except ImportError:

    def pairwise(iterable):
        """ pairwise('ABCDEFG') --> AB BC CD DE EF FG """
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

def triowise(iterable):
    """ triowise('ABCDEFGH') --> ABC BCD CDE DEF EFG FGH """
    b, c = itertools.tee(iterable[1:])
    next(c, None)
    return zip(iterable, b, c)

def check_output_file_exists(output_file: str | bytes | pathlib.PurePath, rewrite: bool=False, clean_run: bool=False) -> pathlib.PurePath:
    """
    Check whether the output file exists. Create the parent directories if they don't exist. Raise FileExistsError in
     case file exists and rewrite is False

    Parameters
    ----------
    output_file : str | bytes | pathlib.PurePath
        path to the output file.
    rewrite :
        whether to rewrite the file if it already exists.
    clean_run :
        whether to print statements while running this function.

    Returns
    ---------
    pathlib.PurePath
        path to the output file
    """
    output_file = pathlib.Path(output_file)
    if not output_file.exists():
        if not output_file.parent.exists():
            output_file.parent.mkdir(parents=True, exist_ok=True)
        return output_file
    if rewrite:
        if not clean_run:
            print(f'file {os.fspath(output_file)} already exists, removing it')
        output_file.unlink()
        return output_file
    else:
        raise FileExistsError(f'{output_file} already exists and rewrite is False')

def _plot_name_helper(plot_output_file_dir, plot_output_file_basename, plot_output_file_suffix):
    if plot_output_file_dir:
        plot_file_name = plot_output_file_basename
        if plot_output_file_suffix:
            plot_file_name += f'_{plot_output_file_suffix}'
        plot_file_name += '.png'
        out_f = plot_output_file_dir / plot_file_name
    else:
        out_f = None
    return out_f

def split_freq_datetime_string(freq_datetime: str) -> Tuple[int | None, str]:
    """ split a datetime frequency into it's integer and string component. E.g.: 30min -> 30, min"""
    if not freq_datetime:
        freq_datetime = '1s'
    match = re.match('([0-9]+)([a-z]+)', freq_datetime, re.I)
    if match:
        items = match.groups()
        return (int(items[0]), items[1])
    return (1, freq_datetime)

class PdTimeDeltaFreq:

    def __init__(self, freq_datetime: str):
        self._freq_dt_int, self._freq_dt_str = split_freq_datetime_string(freq_datetime)
        self._freq_datetime = ''.join([str(self._freq_dt_int), self._freq_dt_str])

    @property
    def freq_datetime(self) -> str:
        return self._freq_datetime

    @property
    def freq_datetime_int(self) -> int:
        return self._freq_dt_int

    @property
    def freq_datetime_str(self) -> str:
        return self._freq_dt_str

def dt_as_pd_datetime(dt: float | int | datetime | np.datetime64 | pd.Timestamp, ref_datetime: datetime | np.datetime64 | pd.Timestamp=sic_def.def_ref_datetime, pd_freq: str | None=None) -> pd.Timestamp:
    """
    Parameters
    ----------
    dt : float | int | datetime | np.datetime64 | pd.Timestamp
        The date, either as datetime object or as seconds from reference datetime
    ref_datetime : datetime | np.datetime64 | pd.Timestamp
        the reference datetime to go from seconds from reference to dates
    pd_freq : str | None
        The pandas reference

    Returns
    -------
    pd.Timestamp
        The datetime as a pandas timestamp
    """
    if dt is not None:
        try:
            float(dt)
        except Exception:
            pass
        else:
            dt = ref_datetime + pd.to_timedelta(dt, unit='second')
        if pd_freq:
            dt = pd.to_datetime(dt).round(freq=pd_freq)
    return dt

def compute_teta(x1: float, y1: float, x2: float, y2: float, x0: float=None, y0: float=None, degrees: bool=False, raise_undetermined_form_error: bool=True, float_atol: float=sic_def.def_float_atol, force_positive: bool=False, clean_run: bool=False) -> float:
    """
    Parameters
    ----------
        x1 : float
        The x coordinate of the first point.
    y1 : float
        The y coordinate of the first point.
    x2 : float
        The x coordinate of the second point.
    y2 : float
        The y coordinate of the second point.
    x0 : float
        The x coordinate of the zero point. Added option to have angles between segments.
    y0 : float
        The y coordinate of the zero point. Added option to have angles between segments.
    degrees : bool
        whether to convert radians to degrees in the result
    float_atol : float
        The float absolute tolerance to consider that points (x0, y0) and (x1, y1) are the same using the function
         math.isclose(). ATTENTION, the value corresponds to abs_tol in the math function!
    raise_undetermined_form_error : bool
        whether to raise FloatingPrecisionError when the values of x and y are the same (0/0 form). Else, slope value
         will be np.nan
    force_positive: bool
        Whether to force the results to always positive. Only used if as_angle is defined. Default is False.
    clean_run : bool
        whether to print statements while running this function.

    Returns
    -------
    float
        the angle formed.
    """
    ang = 'deg' if degrees else 'rad'
    return compute_slope(x1=x1, y1=y1, x2=x2, y2=y2, x0=x0, y0=y0, as_angle=ang, float_atol=float_atol, raise_undetermined_form_error=raise_undetermined_form_error, force_positive=force_positive, clean_run=clean_run)

def compute_slope(x1: float, y1: float, x2: float, y2: float, x0: float=None, y0: float=None, as_angle: None | str=None, raise_zero_division_error: bool=False, raise_undetermined_form_error: bool=True, float_atol: float=sic_def.def_float_atol, force_positive: bool=False, clean_run: bool=False):
    """
    Parameters
    ----------
    x1 : float
        The x coordinate of the first point.
    y1 : float
        The y coordinate of the first point.
    x2 : float
        The x coordinate of the second point.
    y2 : float
        The y coordinate of the second point.
    x0 : float
        The x coordinate of the zero point. Added option to have angles between segments.
    y0 : float
        The y coordinate of the zero point. Added option to have angles between segments.
    as_angle : None | str
        Whether to express the slope as an angle (radians or degrees). Default is None (keeps it as dy / dx).
    float_atol : float
        The float absolute tolerance to consider that points (x0, y0) and (x1, y1) are the same using the function
         math.isclose(). ATTENTION, the value corresponds to abs_tol in the math function!
    raise_zero_division_error : bool
        whether to raise ZeroDivisionError when two consecutive values of x are the same (dy/0 form). Else, slope values
         will be very high/very low.
    raise_undetermined_form_error : bool
        whether to raise FloatingPrecisionError when the values of x and y are the same (0/0 form). Else, slope value
         will be np.nan
    force_positive: bool
        Whether to force the results to always positive. Only used if as_angle is defined. Default is False.
    clean_run : bool
        whether to print statements while running this function.

    Returns
    -------
    float
        The slope / angle formed.
    """
    float_point_error_msg = f'slope in undetermined form 0/0 because points are the same (floating point precision) {x1} == {x2}, \n {y1} == {y2}'
    if x0 is None:
        x0 = x1 + 1.0
    if y0 is None:
        y0 = y1
    v0 = np.array([x0 - x1, y0 - y1])
    v1 = np.array([x2 - x1, y2 - y1])
    if not as_angle:
        close_x, close_y = (False, False)
        if math.isclose(x2, x1, rel_tol=0.0, abs_tol=float_atol):
            msg = f'x the points are the same {x2} == {x1}, (might be due to floating point precision)'
            if raise_zero_division_error:
                raise ZeroDivisionError(msg)
            if not clean_run:
                print(msg)
            close_x = True
            v1[1] = float_atol
        if math.isclose(y2, y1, rel_tol=0.0, abs_tol=float_atol):
            close_y = True
            v1[0] = 0.0
        if close_x and close_y:
            if raise_undetermined_form_error:
                raise FloatingPointError(float_point_error_msg)
            if not clean_run:
                print(float_point_error_msg)
            return np.nan
        return v1[0] / v1[1]
    dot_product = np.dot(v0, v1)
    norm_v0 = np.linalg.norm(v0)
    norm_v1 = np.linalg.norm(v1)
    if norm_v1 == 0.0:
        if raise_undetermined_form_error:
            raise FloatingPointError(float_point_error_msg)
        if not clean_run:
            print(float_point_error_msg)
        return np.nan
    cos_angle = dot_product / (norm_v0 * norm_v1)
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    if not force_positive:
        cross_product = np.cross(v0, v1)
        if cross_product < 0:
            angle_rad = -angle_rad
    if 'deg' in as_angle.lower():
        return np.degrees(angle_rad)
    return angle_rad
if __name__ == '__main__':
    cs_w_test_array = np.array([5.15, 112.0, 113.66, 120.72, 466.78, 492.04, 494.0, 512.0])
    cs_z_test_array = np.array([12.76, 12.78, 13.79, 13.86, 16.09, 16.67, 19.78, 19.91])
    correction_factor = np.nanmean(cs_w_test_array) / np.nanmean(cs_z_test_array)
    cs_z_test_array *= correction_factor
    print(compute_slope(x1=cs_w_test_array[0], y1=cs_z_test_array[0], x2=cs_w_test_array[1], y2=cs_z_test_array[1], as_angle='deg'))