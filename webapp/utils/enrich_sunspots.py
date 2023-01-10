import pandas as pd
import numpy as np
from numpy import hstack
from webapp.utils.trends_util import rolling_mean, find_minimums


def fill_values(data, ndx, func):
    """ fill values from array :attr:`data` using function :attr:`func` """
    vals = [np.ones((ndx[j + 1] - ndx[j])) * func(data[ndx[j]:ndx[j + 1]])
            for j in range(len(ndx) - 1)]
    result = hstack(vals)
    return result


def get_enriched_dataframe(csf_file="data/solarn_month.csv"):
    """
       enrich dataframe with 1y, 3y and 128 months moving averages and
       with min, max and average number of sunspots
    """
    df = pd.read_csv(csf_file, delimiter=";")
    trend = df['sunspots'].values
    # calculate moving average
    df["mean_1y"] = rolling_mean(df['sunspots'], 12)
    df["mean_3y"] = rolling_mean(df['sunspots'], 36)
    df["mean_12y"] = rolling_mean(df['sunspots'], 128)
    # fill the first value of 96.7 instead of NA
    df["mean_1y"] = df["mean_1y"].fillna(96.7)
    df["mean_3y"] = df["mean_3y"].fillna(96.7)
    df["mean_12y"] = df["mean_12y"].fillna(96.7)
    # find minimums in trend using period = 128 months
    mins = find_minimums(trend, 128)
    # correction for short cycle after minimum #7 using period = 119 months
    correction = find_minimums(trend[mins[7]:(mins[7] + 120)], 119)
    # next cycle after minimum #7
    m = mins[7] + correction[1]
    # correction for many zeroes at the end of minimum #5
    k = (mins[5] + mins[6]) // 2
    # drop invalid minimums 6 and 9
    indices = [0] + mins[:5] + [k, mins[7], m, mins[8]] + mins[10:] +\
              [len(trend)]
    # calculate min, max and average number of sunspots for solar cycles
    min_ = fill_values(trend, indices, np.min)
    max_ = fill_values(trend, indices, np.max)
    avg = fill_values(trend, indices, np.mean)
    df["sn_mean"] = pd.Series(avg.tolist())
    df["sn_max"] = pd.Series(max_.tolist())
    df["sn_min"] = pd.Series(min_.tolist())
    return df
