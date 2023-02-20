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
from webapp.config import RNN_INPUT_SIZE, RNN_OUTPUT_SIZE


def load_rnn_model(file_name="models/sn_2lvl_rnn.h5"):
    """ load rnn model """
    model = None
    try:
        model = keras.models.load_model(file_name)
    except IOError:
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
    if len(data) != RNN_INPUT_SIZE:
        raise ValueError("Incorrect input size")
    dat2 = data.reshape(-1, 1, RNN_INPUT_SIZE)
    predict = (np.asarray(SUNSPOTS_MODEL.predict(dat2)))
    result = predict.reshape(RNN_OUTPUT_SIZE)
    new_ts = copy(timeseries[-RNN_OUTPUT_SIZE:])
    res_ts = new_ts + (RNN_OUTPUT_SIZE / 12)
    return result, res_ts


def predict_two_cycles(data, timeseries):
    """ predict next two solar cycles """
    if len(data) != RNN_INPUT_SIZE:
        raise ValueError("Incorrect input size")
    result, res_ts = predict_next_cycle(data, timeseries)
    next_data = hstack([data, result])[-data.shape[0]:]
    next_ts = hstack([timeseries, res_ts])[-data.shape[0]:]
    pred2, pred_ts2 = predict_next_cycle(next_data, next_ts)
    result2 = hstack([next_data, pred2])
    res_ts2 = hstack([res_ts, pred_ts2])
    return result2, res_ts2
