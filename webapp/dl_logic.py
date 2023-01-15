"""
    Apache License 2.0 Copyright (c) 2023 Pavel Bystrov
    deep learning logic
"""
import logging
import numpy as np
import pandas as pd
from copy import copy
from numpy import hstack
from tensorflow import keras


def load_rnn_model(file_name="models/sn_2lvl_rnn.h5"):
    """ load rnn model """
    try:
        model = keras.models.load_model(file_name)
        return model
    except FileNotFoundError:
        logging.error(f"No such file or directory {file_name}")


SUNSPOTS_MODEL = load_rnn_model(file_name="models/sn_2lvl_rnn.h5")


def create_line_plot():
    """ create line plot """
    df = pd.read_csv("data/sunspot_numbers.csv", delimiter=";")
    return df['year_float'].values, df['sunspots'].values


def predict_next_cycle(data, timeseries):
    """ predict next solar cycle """
    dat2 = data.reshape(-1, 1, 1152)
    val_predict2 = (np.asarray(SUNSPOTS_MODEL.predict(dat2)))
    result = val_predict2.reshape(128)
    new_ts = copy(timeseries[-128:])
    res_ts = new_ts + (128 / 12)
    return result, res_ts


def predict_two_cycles(data, timeseries):
    """ predict next solar cycle """
    result, res_ts = predict_next_cycle(data, timeseries)
    next_data = hstack([data, result])[-data.shape[0]:]
    next_ts = hstack([timeseries, res_ts])[-data.shape[0]:]
    result2, res_ts2 = predict_next_cycle(next_data, next_ts)
    return hstack([next_data, result2]), hstack([res_ts, res_ts2])
