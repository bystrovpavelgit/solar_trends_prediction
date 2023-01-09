import numpy as np


def min_index(series, start, interval):
    """ find min index """
    min_ = 300
    index = -1
    for j in range(start, start + interval):
        if j >= len(series):
            break
        if series[j] < min_:
            index = j
            min_ = series[j]
    return index


def find_minimums(series, length, times):
    """ find all minimums in series"""
    k = len(series) // length
    if len(series) > k * length:
        k += 1
    result = [min_index(series, i * length, length) for i in range(k)]
    res = {j: times[j] for j in result}
    return result, res


def moving_average(series, n):
    """
        Calculate average of last n observations
    """
    avg = np.average(series[-n:])
    return avg


def rolling_mean(series, n):
    """
        Calculate average of last n observations
    """
    mean_window_n = series.rolling(window=n).mean()
    return mean_window_n


def exponential_smoothing(series, alpha):
    """
        :func:`exponential_smoothing` - exponential smoothing for data
        :attr:`series` - dataset with timeseries
        :attr:`alpha` - float [0.0, 1.0], smoothing parameter for level
    """
    result = [series[0]]  # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
    return result


def double_exponential_smoothing(series, alpha, beta):
    """
        :func:`double_exponential_smoothing`
        :attr:`series` - dataset with timeseries
        :attr:`alpha` - float [0.0, 1.0], smoothing parameter for level
        :attr:`beta` - float [0.0, 1.0], smoothing parameter for trend
    """
    # first value is same as series
    result = [series[0]]
    level, trend = 0, 0
    for n in range(1, len(series) + 1):
        if n == 1:
            level = series[0]
            trend = series[1] - series[0]
        if n >= len(series):  # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level = level
        level = alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)
    return result


def plot_moving_average(series, window):
    """
        series - dataframe with timeseries
        window - rolling window size
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies

    """
    import matplotlib.pyplot as plt
    mean = series.rolling(window=window).mean()

    plt.figure(figsize=(15, 5))
    plt.title("Moving average\n window size = {}".format(window))
    plt.plot(mean, "g", label="Rolling mean trend")


class HoltWinters:
    """
    :class:`HoltWinters` for the anomalies detection using Brutlag method
    :attr:series - initial time series
    :attr:season_len - length of a season
    :attr:alpha - Holt-Winters model coefficient alpha
    :attr:beta - Holt-Winters model coefficient beta
    :attr:gamma - Holt-Winters model coefficient gamma
    :attr:n_preds - predictions horizon
    :attr:scaling_factor - sets the width of the confidence interval by Brutlag (usually takes values from 2 to 3)
    """

    def __init__(self, series, season_len, alpha, beta, gamma, n_preds,
                 scaling_factor=1.96):
        self.series = series
        self.slen = season_len
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor
        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBond = []
        self.LowerBond = []

    def initial_trend(self):
        sum_ = 0.0
        for i in range(self.slen):
            sum_ += float(self.series[i + self.slen] - self.series[i]) / self.slen
        return sum_ / self.slen

    def initial_seasonal_components(self):
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

    def triple_exponential_smoothing(self):
        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBond = []
        self.LowerBond = []

        seasonals = self.initial_seasonal_components()
        smooth, trend = 0, 0
        for i in range(len(self.series) + self.n_preds):
            if i == 0:  # components initialization
                smooth = self.series[0]
                trend = self.initial_trend()
                self.result.append(self.series[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i % self.slen])

                self.PredictedDeviation.append(0)

                self.UpperBond.append(
                    self.result[0] + self.scaling_factor * self.PredictedDeviation[0]
                )

                self.LowerBond.append(
                    self.result[0] - self.scaling_factor * self.PredictedDeviation[0]
                )
                continue

            if i >= len(self.series):  # predicting
                m = i - len(self.series) + 1
                self.result.append((smooth + m * trend) + seasonals[i % self.slen])

                # when predicting we increase uncertainty on each step
                self.PredictedDeviation.append(self.PredictedDeviation[-1] * 1.01)

            else:
                val = self.series[i]
                last_smooth, smooth = (
                    smooth,
                    self.alpha * (val - seasonals[i % self.slen])
                    + (1 - self.alpha) * (smooth + trend),
                )
                trend = self.beta * (smooth - last_smooth) + (1 - self.beta) * trend
                seasonals[i % self.slen] = (
                        self.gamma * (val - smooth)
                        + (1 - self.gamma) * seasonals[i % self.slen]
                )
                self.result.append(smooth + trend + seasonals[i % self.slen])

                # Deviation is calculated according to Brutlag algorithm.
                self.PredictedDeviation.append(
                    self.gamma * np.abs(self.series[i] - self.result[i])
                    + (1 - self.gamma) * self.PredictedDeviation[-1]
                )

            self.UpperBond.append(
                self.result[-1] + self.scaling_factor * self.PredictedDeviation[-1]
            )

            self.LowerBond.append(
                self.result[-1] - self.scaling_factor * self.PredictedDeviation[-1]
            )

            self.Smooth.append(smooth)
            self.Trend.append(trend)
            self.Season.append(seasonals[i % self.slen])
