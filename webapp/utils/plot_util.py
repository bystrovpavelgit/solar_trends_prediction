"""
    Apache License 2.0 Copyright (c) 2023 Pavel Bystrov
    plot utility
"""
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from matplotlib import pyplot as plt
from webapp.utils.dataframe_util import get_enriched_dataframe


def random_uuid():
    """ returns random uuid """
    res = np.random.randint(10000000)
    return res


def autocorr_image(y, lags=200, figsize=(11, 5), style="bmh"):
    """
        Plot auto-correlation figure and partial auto-correlation figure,
        calculate Dickey–Fuller test
    """

    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    uuid = random_uuid()
    with plt.style.context(style):
        plt.figure(figsize=figsize)
        layout = (2, 1)
        # ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        # ts_ax.plot(y)
        acf_ax = plt.subplot2grid(layout, (0, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 0))

        p_value = sm.tsa.stattools.adfuller(y)[1]
        acf_ax.set_title(
            "Time Series Analysis\n Dickey-Fuller: p={0:.5f}".format(p_value)
        )
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        file_name = f"webapp/static/auto_corr_{uuid}.jpg"
        plt.savefig(file_name)
    return file_name


def prepare_autocorr_data():
    """ prepare auto-correlation data """
    data = get_enriched_dataframe()
    filename = autocorr_image(data["sunspots"])
    print(filename)
    return filename


def timeseries_train_test_split(x, y, test_size):
    """ timeseries train test split """
    # get the index after which test set starts
    test_index = int(len(x) * (1 - test_size))
    x_train = x.iloc[:test_index]
    y_train = y.iloc[:test_index]
    x_test = x.iloc[test_index:]
    y_test = y.iloc[test_index:]
    return x_train, x_test, y_train, y_test


def plot_coefficients(model, x_train):
    """ plot Coefficients """
    coefs = pd.DataFrame(model.coef_, x_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
    return coefs


def code_mean(data, cat_feature, real_feature):
    """ code mean """
    return dict(data.groupby(cat_feature)[real_feature].mean())


def prepare_data(series, lag_start, lag_end, test_size, target_encoding=False):
    """ prepare data """
    # copy of the initial dataset
    data = pd.DataFrame(series.copy())
    data.columns = ["y"]
    # lags of series
    for i in range(lag_start, lag_end):
        data["lag_{}".format(i)] = data.y.shift(i)
    # datetime features
    data.index = pd.to_datetime(data.index)
    data["hour"] = data.index.hour
    data["weekday"] = data.index.weekday
    data["is_weekend"] = data.weekday.isin([5, 6]) * 1
    if target_encoding:
        # calculate averages on train set only
        test_index = int(len(data.dropna()) * (1 - test_size))
        data["weekday_average"] = list(
            map(code_mean(data[:test_index], "weekday", "y").get, data.weekday)
        )
        data["hour_average"] = list(
            map(code_mean(data[:test_index], "hour", "y").get, data.hour)
        )
        # drop encoded variables
        data.drop(["hour", "weekday"], axis=1, inplace=True)
    # train-test split
    y = data.dropna().y
    x = data.dropna().drop(["y"], axis=1)
    x_train, x_test, y_train, y_test = timeseries_train_test_split(
        x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test


def plot_heatmap(ads):
    """ plot heatmap """
    x_train, _, __, ___ = prepare_data(
        ads.Ads, lag_start=6, lag_end=25, test_size=0.3, target_encoding=False
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(x_train.corr())