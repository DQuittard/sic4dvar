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

Created on March 11th 2024 at 10:00
by @Isadora Silva

Last modified on April 18th 2024 at 07:00
by @Isadora Silva

@authors: Isadora Silva
"""
import pathlib
import statistics
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn import metrics
from sic4dvar_functions.helpers.helpers_arrays import arrays_rmv_nan_pair
from sic4dvar_functions.helpers.helpers_plot import helper_plot_scatter

class StatsSummarizer:

    def __init__(self, simulated_data: np.ndarray | Iterable, validation_data: np.ndarray | Iterable):
        simulated_data, validation_data, _ = arrays_rmv_nan_pair(x0=simulated_data, y0=validation_data)
        if simulated_data.size == 0:
            raise IndexError('no data left after removing NaN pair')
        argsort_sim = np.argsort(simulated_data)
        validation_data = validation_data[argsort_sim]
        simulated_data = simulated_data[argsort_sim]
        self._sim, self._val = (simulated_data, validation_data)
        self._df = pd.DataFrame({'sim': self._sim, 'val': self._val})
        self._df['error'] = self._df['val'] - self._df['sim']
        self._df['relative_error'] = self._df['error'] / self._df['val']
        self._validation_mean: float = np.nan
        self._validation_variance: float = np.nan
        self._validation_quantiles: np.array = np.empty(0)
        self._simulation_mean: float = np.nan
        self._simulation_variance: float = np.nan
        self._simulation_quantiles: np.array = np.empty(0)
        self._me: float = np.nan
        self._mre: float = np.nan
        self._mae: float = np.nan
        self._mare: float = np.nan
        self._mse: float = np.nan
        self._msre: float = np.nan
        self._maxe: float = np.nan
        self._maxre: float = np.nan
        self._covariance: float = np.nan
        self._correlation: float = np.nan
        self._r2: float = np.nan
        self._nse: float = np.nan
        self._kge: float = np.nan

    @property
    def df(self):
        return self._df

    @property
    def df_stats(self):
        df_stats = pd.DataFrame(index=['me', 'mae', 'mse', 'rmse', 'maxe', 'mre', 'mare', 'msre', 'rmsre', 'maxre', 'covariance', 'correlation', 'r2', 'nse', 'kge'], columns=['value'])
        for i in df_stats.index:
            df_stats.loc[i, 'value'] = self.__getattribute__(i)
        df_stats = df_stats.round(3)
        return df_stats

    @property
    def validation_mean(self):
        """ the mean of the validation values """
        if np.isnan(self._validation_mean):
            self._validation_mean = np.nanmean(self._val)
        return self._validation_mean

    @property
    def validation_variance(self):
        """ the variance of the validation values """
        if np.isnan(self._validation_variance):
            self._validation_variance = np.nanmean((self._val - self.validation_mean) ** 2)
        return self._validation_variance

    @property
    def validation_std(self):
        """ the standard deviation of the validation values """
        return np.sqrt(self.validation_variance)

    @property
    def validation_quantiles(self):
        """ min, 1%, 10%, 25%, 50%, 75%, 90%, 99% and max of simulation values """
        if self._validation_quantiles.size == 0:
            self._validation_quantiles = np.nanquantile(self._val, [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
        return self._validation_quantiles

    @property
    def simulation_mean(self):
        """ the mean of the simulation values """
        if np.isnan(self._simulation_mean):
            self._simulation_mean = np.nanmean(self._sim)
        return self._simulation_mean

    @property
    def simulation_variance(self):
        """ the variance of the simulation values """
        if np.isnan(self._simulation_variance):
            self._simulation_variance = np.nanmean((self._sim - self.simulation_mean) ** 2)
        return self._simulation_variance

    @property
    def simulation_std(self):
        """ the standard deviation of the simulation values """
        return np.sqrt(self.simulation_variance)

    @property
    def simulation_quantiles(self):
        """ min, 1%, 10%, 25%, 50%, 75%, 90%, 99% and max of simulation values """
        if self._simulation_quantiles.size == 0:
            self._simulation_quantiles = np.nanquantile(self._sim, [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
        return self._simulation_quantiles

    @property
    def me(self):
        """
        ME (Mean Error) is the average of the differences between the true and predicted values
        """
        if np.isnan(self._me):
            self._me = np.nanmean(self._df['error'])
        return self._me

    @property
    def mre(self):
        """
        MRE (Mean Relative Error) is the average of the differences between the true and predicted values relative
         to the true values
        """
        if np.isnan(self._mre):
            self._mre = np.nanmean(self._df['relative_error'])
        return self._mre

    @property
    def mae(self):
        """
        MAE (Mean Absolute Error) is the average of the absolute differences between the true and predicted values
        """
        if np.isnan(self._mae):
            self._mae = np.nanmean(np.abs(self._df['error']))
        return self._mae

    @property
    def mare(self):
        """
        MAE (Mean Absolute Relative Error) is the average of the absolute differences between the true and predicted
         values relative to the true values
        """
        if np.isnan(self._mare):
            self._mare = np.nanmean(np.abs(self._df['relative_error']))
        return self._mare

    @property
    def mse(self):
        """
        MSE (Mean Squared Error) is the average of the squared differences between the true and predicted values
        """
        if np.isnan(self._mse):
            self._mse = np.nanmean(self._df['error'] ** 2)
        return self._mse

    @property
    def msre(self):
        """
        MSE (Mean Squared Relative Error) is the average of the squared differences between the true and predicted
         values relative to the true values
        """
        if np.isnan(self._msre):
            self._msre = np.nanmean(self._df['error'] ** 2 / self._df['val'] ** 2)
        return self._msre

    @property
    def rmse(self):
        """
        RMSE (Root Mean Squared Error) is the square root of the average of the squared differences between the true
         and predicted values
        """
        return np.sqrt(self.mse)

    @property
    def rmsre(self):
        """
        RMSE (Root Mean Squared Relative Error) is the square root of the average of the squared differences between
        the true and predicted values relative to the true values
        """
        return np.sqrt(self.msre)

    @property
    def maxe(self):
        """
        Max Error is the maximum residual error, a metric that captures the worst case error between the predicted
         values and the true values.
        """
        if np.isnan(self._maxe):
            self._maxe = np.nanmax(np.abs(self._df['error']))
        return self._maxe

    @property
    def maxre(self):
        """
        Max Relative Error is the maximum residual error, a metric that captures the worst case error between the
         predicted values and the true values relative to the true values.
        """
        if np.isnan(self._maxre):
            self._maxre = np.nanmax(np.abs(self._df['relative_error']))
        return self._maxre

    @property
    def covariance(self):
        """
        Covariance is a measure of the joint variability of two variables.
        Return the sample covariance.
        """
        if np.isnan(self._covariance):
            self._covariance = statistics.covariance(self._sim, self._val)
        return self._covariance

    @property
    def correlation(self):
        """
        Pearson's correlation coefficient. Return values between -1 and +1.
        It measures the strength and direction of the linear relationship, where +1 means very strong, positive linear
         relationship, -1 very strong, negative linear relationship, and 0 no linear relationship.
        """
        if np.isnan(self._correlation):
            self._correlation = statistics.correlation(self._sim, self._val)
        return self._correlation

    @property
    def r2(self):
        """
        R squared (Coefficient of determination), is the proportion of the variation in the dependent variable that is
         predictable from the independent variable(s).
        As such variance is dataset dependent, may not be meaningfully comparable across different datasets. Best
         possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model
         that always predicts the expected (average) value of y, disregarding the input features, would get a score
         of 0.0.
        """
        if np.isnan(self._r2):
            self._r2 = metrics.r2_score(self._sim, self._val)
        return self._r2

    @property
    def nse(self):
        """
        The Nash–Sutcliffe efficiency (NSE) is calculated as one minus the ratio of the error variance of the predicted
         values divided by the variance of the true values. In the situation of a perfect model with an
         estimation error variance equal to zero, the resulting Nash–Sutcliffe Efficiency equals 1 (NSE = 1)
        """
        if np.isnan(self._nse):
            nse_num = np.nansum(self._df['error'] ** 2)
            nse_den = np.nansum((self._val - self.validation_mean) ** 2)
            self._nse = 1 - nse_num / nse_den
        return self._nse

    @property
    def kge(self):
        """
        The Kling-Gupta Efficiency  https://hess.copernicus.org/preprints/hess-2019-327/hess-2019-327.pdf
        """
        if np.isnan(self._kge):
            r = self.correlation
            alfa = self.simulation_mean / self.validation_mean
            beta = self.simulation_variance / self.validation_variance
            kse_cor = r - 1
            kse_var = alfa - 1
            kse_bias = beta - 1
            self._kge = 1 - np.sqrt(kse_cor ** 2 + kse_var ** 2 + kse_bias ** 2)
        return self._kge

    def plot(self, show: bool=True, title: str='Sim x Val', x_axis_title: str='simulation', y_axis_title: str='validation', x_lim: Iterable | None=None, y_lim: Iterable | None=None, fig_width: int=5, fig_height: int=5, add_params: tuple=('r2', 'correlation'), output_file: str | bytes | pathlib.PurePath | None=None):
        add_text_bottom_right = ''
        if add_params:
            for k in add_params:
                v = getattr(self, k)
                add_text_bottom_right += '\n\t{}: {:.3f}'.format(k, v)
        helper_plot_scatter(xs=[self._sim], ys=[self._val], show=show, title=title, x_axis_title=x_axis_title, y_axis_title=y_axis_title, x_lim=x_lim, y_lim=y_lim, fig_width=fig_width, fig_height=fig_height, add_identity_line=True, add_text_bottom_right=add_text_bottom_right, output_file=output_file)

    def __repr__(self):
        basestr = 'StatsSummarizer('
        for k in ['validation_mean', 'validation_std', 'simulation_mean', 'simulation_std', 'me', 'mae', 'mse', 'rmse', 'maxe', 'mre', 'mare', 'msre', 'rmsre', 'maxre', 'covariance', 'correlation', 'r2', 'nse', 'kge']:
            v = getattr(self, k)
            basestr += '\n\t{:16}: {:.3f},'.format(k, v)
        basestr += ')'
        return basestr