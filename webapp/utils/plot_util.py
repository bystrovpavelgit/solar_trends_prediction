"""
    Apache License 2.0 Copyright (c) 2023 Pavel Bystrov
    plot utility
"""
import logging
import os
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from matplotlib import pyplot as plt
from webapp.utils.dataframe_util import get_enriched_dataframe, prepare_data


def random_uuid():
    """ returns random uuid """
    res = np.random.randint(10000000)
    return res


def autocorr_image(y, lags=200, figsize=(11, 5), style="bmh"):
    """
        Plot auto-correlation figure and partial auto-correlation figure,
        calculate Dickey–Fuller test
    """
    if y is None:
        raise ValueError("Input timeseries are empty")

    uuid = random_uuid()
    file_name = os.path.join("webapp", "static", f"auto_corr_{uuid}.jpg")
    with plt.style.context(style):
        plt.figure(figsize=figsize)
        layout = (2, 1)
        acf_ax = plt.subplot2grid(layout, (0, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 0))

        p_value = sm.tsa.stattools.adfuller(y)[1]
        acf_ax.set_title(
            "Time Series Analysis\n Dickey-Fuller: p={0:.5f}".format(p_value)
        )
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)  # LinAlgError
        plt.tight_layout()
        try:
            os.makedirs("webapp/static", exist_ok=True)
            plt.savefig(file_name)
        except FileNotFoundError as exc:
            # numpy.linalg.LinAlgError: Singular matrix
            msg = f"Directory for file {file_name} not exists"
            logging.error(msg)
            raise exc
    return file_name, p_value


def prepare_autocorr_data():
    """ prepare auto-correlation data """
    data = get_enriched_dataframe()
    filename = autocorr_image(data["sunspots"])
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
    uuid = random_uuid()
    file_name = os.path.join("webapp", "static", f"heatmap_{uuid}.jpg")
    try:
        plt.figure(figsize=(11, 7))
        layout = (1, 1)
        plt.subplot2grid(layout, (0, 0))
        os.makedirs("webapp/static", exist_ok=True)
        sns.heatmap(correlations).get_figure().savefig(file_name)
    except FileNotFoundError as exc:
        msg = f"Directory for file {file_name} not exists"
        logging.error(msg)
        raise exc
    return file_name
