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


def find_minimum_with_strategy(data: array, start: int, length: int,
                               inc: int = 40, strategy: int = 1) -> int:
    """
       find minimums using provided strategy
       :attr:`strategy` = 0 - check the first half for minimum
       :attr:`strategy` = 1 - check the second half for minimum
    """
    if data is None:
        raise ValueError("Input array is empty")
    if len(data) < start or len(data) < length:
        raise ValueError("Input array is too short")
    size = len(data)
    # intervals
    intv = [((i * length) // 2) + start for i in range(2)] + \
           [min(start + length + inc, size)]
    # indices from data array
    ndx = list(np.arange(0, start + length + inc))
    # minimums
    minimums = [np.min(data[intv[i]: intv[i + 1]]) for i in range(2)]
    # data from the first half and from the second half
    splits = [data[intv[i]: intv[i + 1]] for i in range(2)]
    # indices of minimums from splits (-1 placed for non minimums)
    min_indices = [np.where(splits[i] == minimums[i],
                            ndx[intv[i]: intv[i + 1]], -1) for i in range(2)]
    if strategy == 0:
        result = int(list(filter(lambda x: x > -1,
                                 min_indices[0].tolist()))[0])
        return result
    result = int(list(filter(lambda x: x > -1, min_indices[1].tolist()))[0])
    return result


def get_all_minimums(data: array, length: int = 128) -> list:
    """ find all minimums in data """
    if data is None or len(data) == 0:
        return []
    if length <= 0 or len(data) < length:
        return []
    size = len(data)
    minimums = []
    init = 0
    while (init + (length // 2)) < size:
        if init == 0:
            result0 = find_minimum_with_strategy(data, 0, length, strategy=0)
            result1 = find_minimum_with_strategy(data, 0, length, strategy=1)
            if data[result0] < data[result1]:
                res = result0
            else:
                res = result1
        else:
            res = find_minimum_with_strategy(data, init, length, strategy=1)
        init = res + 1
        minimums.append(res)
    return minimums


def rolling_mean(series: Series, num: int) -> Series:
    """
        Calculate average of last nun observations
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
    """ returns sunspot numbers data from csv-file if file exists"""
    try:
        data = pd.read_csv(csv_file, delimiter=";")
        year_float = data["year_float"].values
        sunspots = data["sunspots"].values
        return Optional.of((year_float, sunspots))
    except FileNotFoundError:
        message = f"File {csv_file} not found"
        logging.error(message)
        return Optional.empty()


def get_enriched_dataframe(csv_file: str = "data/sunspot_numbers.csv") \
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
    ndxes = [0] + get_all_minimums(trend) + [len(trend)]
    # calculate min, max and average number of sunspots for solar cycles
    min_ = fill_values(trend, ndxes, np.min)
    max_ = fill_values(trend, ndxes, np.max)
    avg = fill_values(trend, ndxes, np.mean)
    data["sn_mean"] = pd.Series(avg.tolist())
    data["sn_max"] = pd.Series(max_.tolist())
    data["sn_min"] = pd.Series(min_.tolist())
    y_max = hstack([np.zeros([ndxes[17]]),
                    np.ones((ndxes[20] - ndxes[17])),
                    np.zeros([ndxes[-1] - ndxes[20]])])
    y_min = hstack([np.zeros([ndxes[5]]),
                    np.ones((ndxes[8] - ndxes[5])),
                    np.zeros([ndxes[-1] - ndxes[8]])])
    data["y_min"] = pd.Series(y_min.tolist())
    data["y_max"] = pd.Series(y_max.tolist())
    return data


def prepare_data(csv_file="data/sunspot_numbers.csv",
                 lag_start=1,
                 lag_end=26):
    """ prepare data """
    try:
        data = pd.read_csv(csv_file, delimiter=";")
        fields = []
        for i in range(lag_start, (lag_end + 1)):
            data[f"lag_{i}"] = data["sunspots"].shift(i).fillna(0.)
            fields.append(f"lag_{i}")
        for i in range(64, 514, 64):
            data[f"lag_{i}"] = data["sunspots"].shift(i).fillna(0.)
            fields.append(f"lag_{i}")
    except FileNotFoundError as exc:
        logging.error("File data/sunspot_numbers.csv not found")
        raise exc
    return data
