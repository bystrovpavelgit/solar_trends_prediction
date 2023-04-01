"""
    Apache License 2.0 Copyright (c) 2023 Pavel Bystrov
    plot utility
"""
import numpy as np
import os
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from matplotlib import pyplot as plt
from webapp.utils.dataframe_util import get_enriched_dataframe, \
    prepare_data


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


def get_lag_fields():
    """ get lag fields """
    names = [f"lag_{num}" for num in range(1, 27)]
    names += [f"lag_{num}" for num in range(64, 514, 64)]
    return names


def plot_lags_correlation_heatmap() -> str:
    """ plot heatmap """
    dataframe = prepare_data()
    correlations = dataframe[get_lag_fields()].corr()
    os.makedirs("webapp/static", exist_ok=True)
    uuid = random_uuid()
    file_name = os.path.join("webapp", "static", f"heatmap_{uuid}.jpg")
    sns.heatmap(correlations).get_figure().savefig(file_name)
    return file_name
