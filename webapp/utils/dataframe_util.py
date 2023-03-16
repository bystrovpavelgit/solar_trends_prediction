"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    utility to handle dataframes
"""
import logging
import numpy as np
import pandas as pd
from numpy import hstack, array
from pandas import Series
from optional import Optional


def min_index(series: array, start: int, interval: int) -> int:
    """ find min index """
    index = -1
    if len(series) == 0:
        return index
    min_ = np.max(series) + 1.
    for j in range(start, min(start + interval, len(series))):
        if series[j] < min_:
            index = j
            min_ = series[j]
    return index


def find_minimums(series: array, length: int) -> list:
    """ find all minimums in series """
    if length <= 0:
        return []
    k = len(series) // length
    if len(series) > k * length:
        k += 1
    result = [min_index(series, i * length, length) for i in range(k)]
    return result


def rolling_mean(series: Series, num: int) -> Series:
    """
        Calculate average of last n observations
    """
    mean_window_n = series.rolling(window=num).mean()
    return mean_window_n


def fill_values(data, ndx, func):
    """ fill values from array :attr:`data` using function :attr:`func` """
    if len(data) < 2:
        return data
    for index_ in ndx:
        if index_ > len(data):
            return np.array([])
    vals = [np.ones((ndx[j + 1] - ndx[j])) * func(data[ndx[j]:ndx[j + 1]])
            for j in range(len(ndx) - 1)]
    result = hstack(vals)
    return result


def sunspot_numbers(csv_file: str = "data/sunspot_numbers.csv") -> Optional:
    """ returns sunspot numbers data """
    try:
        data = pd.read_csv(csv_file, delimiter=";")
        year_float = data["year_float"].values
        sunspots = data["sunspots"].values
        return Optional.of((year_float, sunspots))
    except FileNotFoundError:
        message = f"File {csv_file} not found"
        logging.error(message)
        return Optional.empty()


def get_users_timeseries(csv_file: str = "data/hour_online.csv") -> Optional:
    """
       enrich users timeseries with 12 points moving average,
       36 points moving averages and 128 points moving average
    """
    try:
        times = pd.read_csv(csv_file)
        # fill the first value of 34002 instead of NA
        times["mean_12p"] = rolling_mean(times["Users"], 12).fillna(34002)
        times["mean_36p"] = rolling_mean(times["Users"], 36).fillna(34002)
        times["mean_128p"] = rolling_mean(times["Users"], 128).fillna(34002)
        return Optional.of(times)
    except FileNotFoundError:
        message = f"File not found {csv_file}"
        logging.error(message)
        return Optional.empty()


def get_enriched_dataframe(csv_file: str = "data/sunspot_numbers.csv")\
        -> pd.DataFrame:
    """
       enrich dataframe with 1y, 3y and 128 months moving averages and
       with min, max and average number of sunspots
    """
    data = pd.read_csv(csv_file, delimiter=";")
    trend = data["sunspots"].values
    # calculate moving average
    data["mean_1y"] = rolling_mean(data["sunspots"], 12)
    data["mean_3y"] = rolling_mean(data["sunspots"], 36)
    data["mean_12y"] = rolling_mean(data["sunspots"], 128)
    # fill the first value of 96.7 instead of NA
    data["mean_1y"] = data["mean_1y"].fillna(96.7)
    data["mean_3y"] = data["mean_3y"].fillna(96.7)
    data["mean_12y"] = data["mean_12y"].fillna(96.7)
    # find minimums in trend using period = 128 months
    mins = find_minimums(trend, 128)
    # correction for short cycle after minimum #7 using period = 119 months
    correction = find_minimums(trend[mins[7]:(mins[7] + 120)], 119)
    # next cycle after minimum #7
    cy8 = mins[7] + correction[1]
    # correction for many zeroes at the end of minimum #5
    cy6 = (mins[5] + mins[6]) // 2
    # drop invalid minimums 6 and 9
    indices = [0] + mins[:5] + [cy6, mins[7], cy8, mins[8]] + mins[10:] + \
              [len(trend)]
    # calculate min, max and average number of sunspots for solar cycles
    min_ = fill_values(trend, indices, np.min)
    max_ = fill_values(trend, indices, np.max)
    avg = fill_values(trend, indices, np.mean)
    data["sn_mean"] = pd.Series(avg.tolist())
    data["sn_max"] = pd.Series(max_.tolist())
    data["sn_min"] = pd.Series(min_.tolist())

    y_max = hstack([np.zeros([indices[17]]),
                    np.ones((indices[20] - indices[17])),
                    np.zeros([indices[-1] - indices[20]])])
    y_min = hstack([np.zeros([indices[5]]),
                    np.ones((indices[8] - indices[5])),
                    np.zeros([indices[-1] - indices[8]])])
    data["y_min"] = pd.Series(y_min.tolist())
    data["y_max"] = pd.Series(y_max.tolist())
    return data


def prepare_data(csv_file="data/sunspot_numbers.csv",
                 lag_start=1,
                 lag_end=24):
    """ """
    data = pd.read_csv(csv_file, delimiter=";")
    # lags of series
    for i in range(lag_start, (lag_end + 1)):
        data[f"lag_{i}"] = data.sunspots.shift(i).fillna(0.)
    final_lag = 12
    for i in range(3, (final_lag + 1)):
        data[f"lag_{i * 12}"] = data.sunspots.shift(i * 12).fillna(0.)
    return data
