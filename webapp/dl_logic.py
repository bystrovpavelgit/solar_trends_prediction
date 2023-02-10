"""
    Apache License 2.0 Copyright (c) 2023 Pavel Bystrov
    deep learning logic
"""
import logging
from copy import copy
import numpy as np
import pandas as pd
from numpy import hstack
from tensorflow import keras


def load_rnn_model(file_name="models/sn_2lvl_rnn.h5"):
    """ load rnn model """
    model = None
    try:
        model = keras.models.load_model(file_name)
    except FileNotFoundError:
        msg = f"No such file or directory {file_name}"
        logging.error(msg)
    return model


SUNSPOTS_MODEL = load_rnn_model(file_name="models/sn_2lvl_rnn.h5")


def sunspot_numbers():
    """ returns sunspot numbers data """
    data = pd.read_csv("data/sunspot_numbers.csv", delimiter=";")
    year_float = data['year_float'].values
    sunspots = data['sunspots'].values
    return year_float, sunspots


def predict_next_cycle(data, timeseries):
    """ predict next solar cycle """
    dat2 = data.reshape(-1, 1, 1152)
    val_predict2 = (np.asarray(SUNSPOTS_MODEL.predict(dat2)))
    result = val_predict2.reshape(128)
    new_ts = copy(timeseries[-128:])
    res_ts = new_ts + (128 / 12)
    return result, res_ts


def predict_two_cycles(data, timeseries):
    """ predict next two solar cycles """
    result, res_ts = predict_next_cycle(data, timeseries)
    next_data = hstack([data, result])[-data.shape[0]:]
    next_ts = hstack([timeseries, res_ts])[-data.shape[0]:]
    pred2, pred_ts2 = predict_next_cycle(next_data, next_ts)
    result2 = hstack([next_data, pred2])
    res_ts2 = hstack([res_ts, pred_ts2])
    return result2, res_ts2
