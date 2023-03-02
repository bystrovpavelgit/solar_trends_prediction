"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    trends modification utility
"""
from functools import partial
import numpy as np
from numpy import array, fft
from pandas import Series
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


def exponential_smoothing(series, alpha):
    """
        :func:`exponential_smoothing` - exponential smoothing for data
        :attr:`series` - dataset with timeseries
        :attr:`alpha` - float [0.0, 1.0], smoothing parameter for level
    """
    if alpha < 0:
        raise ValueError("alpha and beta must be positive")
    if series is None or len(series) == 0:
        raise ValueError("Input array must be non empty")
    result = [series[0]]  # first value is same as series
    for num in range(1, len(series)):
        result.append(alpha * series[num] + (1 - alpha) * result[num - 1])
    return result


def double_exponential_smoothing(series, alpha, beta):
    """
        :func:`double_exponential_smoothing`
        :attr:`series` - dataset with timeseries
        :attr:`alpha` - float [0.0, 1.0], smoothing parameter for level
        :attr:`beta` - float [0.0, 1.0], smoothing parameter for trend
    """
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be positive")
    if series is None or len(series) == 0:
        raise ValueError("Input array must be non empty")
    # first value is same as series
    result = [series[0]]
    level, trend = 0, 0
    for num in range(1, len(series) + 1):
        if num == 1:
            level = series[0]
            trend = series[1] - series[0]
        if num >= len(series):  # forecasting
            value = result[-1]
        else:
            value = series[num]
        last_level = level
        level = alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)
    return result


class HoltWinters:
    """
    :class:`HoltWinters` for the anomalies detection using Brutlag method
    :attr:series - initial time series
    :attr:season_len - length of a season
    :attr:alpha - Holt-Winters model coefficient alpha
    :attr:beta - Holt-Winters model coefficient beta
    :attr:gamma - Holt-Winters model coefficient gamma
    :attr:n_preds - predictions horizon
    :attr:scaling_factor - sets the width of the confidence interval by Brutlag
                (usually takes values from 2 to 3)
    """
    def __init__(self, series, season_len, alpha, beta, gamma, n_preds,
                 scaling_factor=1.96):
        """ init """
        if n_preds < 0:
            raise ValueError("n_preds should be positive")
        self.series = series
        self.slen = season_len
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor
        self.result = []
        self.Smooth = []
        self.seasons = []
        self.trends = []
        self.predicted_deviation = []
        self.upper_bond = []
        self.lower_bond = []

    def initial_trend(self):
        """ initial trend """
        sum_ = 0.0
        for i in range(self.slen):
            sum_ += float(self.series[i + self.slen] - self.series[i]) / \
                    self.slen
        res = sum_ / self.slen
        return res

    def initial_seasonal_components(self):
        """ initial_seasonal_components """
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.series) / self.slen)
        # let's calculate season averages
        for j in range(n_seasons):
            season_averages.append(
                sum(self.series[self.slen * j: self.slen * j + self.slen])
                / float(self.slen)
            )
        # let's calculate initial values
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += (
                        self.series[self.slen * j + i] - season_averages[j]
                )
            seasonals[i] = sum_of_vals_over_avg / n_seasons
        return seasonals

    def initialize_params(self):
        """
           initialize result, Smooth, seasons, trends, predicted_deviation,
           upper_bond, and lower_bond params
        """
        self.result = []
        self.Smooth = []
        self.seasons = []
        self.trends = []
        self.predicted_deviation = []
        self.upper_bond = []
        self.lower_bond = []

    def components_initialization(self, seasonals_):
        """
           components initialization of result, Smooth, seasons, trends,
           predicted_deviation, upper_bond, and lower_bond
        """
        trend_ = self.initial_trend()
        self.predicted_deviation = [0]
        predict = self.scaling_factor * self.predicted_deviation[0]
        self.result = [self.series[0]]
        self.Smooth = [self.series[0]]
        self.trends = [trend_]
        self.seasons = [seasonals_[0]]
        self.upper_bond = [self.result[0] + predict]
        self.lower_bond = [self.result[0] - predict]

    def triple_exponential_smoothing(self):
        """ triple exponential smoothing """
        if len(self.series) + self.n_preds == 0:
            self.initialize_params()
            return self.result
        seasonals = self.initial_seasonal_components()
        self.components_initialization(seasonals)
        smooth = self.series[0]
        trend = self.initial_trend()
        for i in range(1, len(self.series) + self.n_preds):
            if i >= len(self.series):
                num = i - len(self.series) + 1
                self.result.append((smooth + num * trend) +
                                   seasonals[i % self.slen])
                # when predicting we increase uncertainty on each step
                self.predicted_deviation.append(
                    self.predicted_deviation[-1] * 1.01)
            else:
                val = self.series[i]
                last_smooth, smooth = (
                    smooth,
                    self.alpha * (val - seasonals[i % self.slen])
                    + (1 - self.alpha) * (smooth + trend),
                )
                delta_smooth = smooth - last_smooth
                trend = self.beta * delta_smooth + (1 - self.beta) * trend
                seasonals[i % self.slen] = (
                        self.gamma * (val - smooth)
                        + (1 - self.gamma) * seasonals[i % self.slen]
                )
                self.result.append(smooth + trend + seasonals[i % self.slen])
                # Deviation is calculated according to Brutlag algorithm.
                self.predicted_deviation.append(
                    self.gamma * np.abs(self.series[i] - self.result[i])
                    + (1 - self.gamma) * self.predicted_deviation[-1]
                )
            predict = self.scaling_factor * self.predicted_deviation[-1]
            self.upper_bond.append(self.result[-1] + predict)
            self.lower_bond.append(self.result[-1] - predict)
            self.Smooth.append(smooth)
            self.trends.append(trend)
            self.seasons.append(seasonals[i % self.slen])
        return self.result


def timeseries_cv_score(values, params, cycle=128):
    """ timeseries cross-validation score """
    if cycle <= 0:
        raise ValueError("cycle must be positive")
    if len(values) == 0 or len(params) == 0:
        raise ValueError("Input arrays must be non empty")
    errors = []
    tscv = TimeSeriesSplit(n_splits=3)
    # строим прогноз на отложенной выборке и считаем ошибку
    for train, test in tscv.split(values):
        model = HoltWinters(
            series=values[train],
            season_len=cycle,
            alpha=params[0],
            beta=params[1],
            gamma=params[2],
            n_preds=len(test),
        )
        model.triple_exponential_smoothing()
        predictions = model.result[-len(test):]
        actual = values[test]
        error = mean_squared_error(predictions, actual)
        errors.append(error)
    mse = np.mean(np.array(errors))
    return mse


def get_optimal_params(data):
    """ функция для нахождения оптимальных параметров классификаторов """
    args = np.array([0, 0, 0])
    if data is None or len(data) == 0:
        return args.tolist()
    timeseries_cv_func = partial(timeseries_cv_score, data.values)
    # минимизируем L
    opt = minimize(timeseries_cv_func, x0=args, method="TNC",
                   bounds=((0, 1), (0, 1), (0, 1)))
    result = opt.x
    return result


def hw_exponential_smoothing(data: Series, sess_len: int = 128) -> list:
    """ triple exponential smoothing using Holt-Winters model """
    if sess_len <= 0:
        raise ValueError("sess_len must be positive")
    if data is None or len(data) == 0:
        raise ValueError("Input array must be non empty")
    opt_params = get_optimal_params(data)
    model = HoltWinters(
        data[:-sess_len],
        season_len=sess_len,
        alpha=opt_params[0],
        beta=opt_params[1],
        gamma=opt_params[2],
        n_preds=sess_len,
        scaling_factor=2.56)
    predictions = model.triple_exponential_smoothing()
    return predictions


def get_fourier_prediction(x_data: array, times: array,
                           n_predict: int, n_harm: int = 120) -> tuple:
    """ calculate fourier amplitudes using numpy.fft """
    if n_predict <= 0 or n_harm <= 0:
        raise ValueError("n_predict and n_harm must be positive")
    if len(x_data) == 0 or len(times) == 0:
        raise ValueError("Input arrays must be non empty")
    time = np.arange(0, x_data.size)
    poly = np.polyfit(time, x_data, 1)
    x_freqdom = fft.fft(x_data - poly[0] * time)
    freq = fft.fftfreq(x_data.size)
    indexes = list(range(x_data.size))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key=lambda inp: np.absolute(freq[inp]))
    time = np.arange(0, x_data.size + n_predict)
    restored_sig = np.zeros(time.size)
    for i in indexes[:1 + n_harm * 2]:
        amplitude = np.absolute(x_freqdom[i]) / x_data.size
        phase = np.angle(x_freqdom[i])
        restored_sig += \
            amplitude * np.cos(2 * np.pi * freq[i] * time + phase)
    res = restored_sig + poly[0] * time
    tuple_ = (res[-n_predict:], times[-n_predict:] + (n_predict / 12))
    return tuple_
