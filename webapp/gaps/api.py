"""
    Apache License 2.0 Copyright (c) 2023 Pavel Bystrov
    api func for gap views
"""
import numpy as np
from pandas import Series
from scipy import stats
from webapp.utils.dataframe_util import get_all_minimums


def count_gaps(data: Series, gap_marker: float = -1.0) -> int:
    """ count gaps in data """
    cnt = 0
    for val in data.values:
        if val == gap_marker:
            cnt += 1
    return cnt


def confidence_interval(std, alpha, num):
    """ confidence_interval """
    if std < 0 or alpha <= 0 or num <= 0:
        raise ValueError("Incorrect arguments std, alpha, and number")
    z_a = stats.t.ppf(1 - alpha * .5, num - 1)
    interval = std * z_a / np.sqrt(num)
    return interval


def describe_sunspots_intervals(data: np.array) -> tuple:
    """ describe solar intervals using t-statistics """
    lst = get_all_minimums(data, 128)
    cycles = [lst[i] - lst[i - 1] for i in range(1, len(lst))]
    num = len(cycles)
    cycles = np.array(sorted(cycles))
    std = cycles.std() / 12
    x_mean = cycles.mean() / 12
    mean_msg = f"Solar cycle mean duration is {x_mean} years " \
               f"with std. deviation {std} years"
    alpha = 0.006
    x_intv = confidence_interval(std, alpha, num)
    msg2 = f"99.4 % Confidence interval for solar cycle duration is" \
           f" [{x_mean - x_intv}, {x_mean + x_intv}] (years)"
    # min amount
    z_a = stats.norm.ppf(0.995)
    st_deviation, interval = 1.4, 1.1
    num = (z_a * st_deviation / interval) ** 2
    num = int(num + 0.99)  # top
    msg3 = f"Min sequence length N is {num} for" \
           " sigma = 1.4, alpha = 0.01 and interval = 1.1"
    return cycles, mean_msg, msg2, msg3
