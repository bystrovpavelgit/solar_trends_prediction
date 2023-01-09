import pandas as pd
import numpy as np
from numpy import hstack
from webapp import create_app
from webapp.db import DB
from webapp.utils.trends_util import rolling_mean, find_minimums


def fill_values(data, ndx, func):
    """ fill values from array data using function func """
    vals = [np.ones((ndx[j + 1] - ndx[j])) * func(data[ndx[j]:ndx[j + 1]])
            for j in range(len(ndx) - 1)]
    result = hstack(vals)
    return result
# plt.figure(figsize=(16, 5))
# plt.plot(df['year_float'].values, max_)
# plt.plot(df['year_float'].values, df["mean_1y"].values)
# plt.plot(df['year_float'].values, df["mean_12y"].values)
# plt.plot(df['year_float'].values, avg)
# plt.plot(df['year_float'].values, min_)
# plt.show()


if __name__ == "__main__":
    app = create_app()
    with app.app_context():
        df = pd.read_csv("../data/solarn_month.csv", delimiter=";")
        df["mean_1y"] = rolling_mean(df['sunspots'], 12)
        df["mean_3y"] = rolling_mean(df['sunspots'], 36)
        df["mean_12y"] = rolling_mean(df['sunspots'], 128)
        df["mean_1y"] = df["mean_1y"].fillna(96.7)
        df["mean_3y"] = df["mean_3y"].fillna(96.7)
        df["mean_12y"] = df["mean_12y"].fillna(96.7)

        data = df['sunspots'].values
        lst, tup = find_minimums(data, 128, df['year_float'].values)
        lll, _ = find_minimums(data[lst[7]:(lst[7] + 120)], 119, df['year_float'].values[lst[7]:(lst[7] + 120)])
        k = (lst[5] + lst[6]) // 2
        m = lst[7] + lll[1]
        lst = lst[:5] + [k, lst[7], m, lst[8]] + lst[10:]
        indx = [0] + lst + [len(data)]

        min_ = fill_values(data, indx, np.min)
        max_ = fill_values(data, indx, np.max)
        avg = fill_values(data, indx, np.mean)
        df["sn_mean"] = pd.Series(avg.tolist())
        df["sn_max"] = pd.Series(max_.tolist())
        df["sn_min"] = pd.Series(min_.tolist())
