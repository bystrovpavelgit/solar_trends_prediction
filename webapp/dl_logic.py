"""
    Apache License 2.0 Copyright (c) 2023 Pavel Bystrov
    deep learning logic
"""
import logging
from copy import copy
import numpy as np
from numpy import hstack
from sklearn.metrics import mean_absolute_error
from tensorflow import keras
from tensorflow.keras import models, layers
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
LAGS_DNN_MODEL = load_rnn_model(file_name="models/two_layer_dnn.h5")


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


def get_two_layers_nnet():
    """ prepare two layers neural net """
    model = models.Sequential()
    model.add(layers.Dense(34, activation="tanh", input_shape=(34,)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(12, activation="tanh"))
    model.add(layers.Dense(1))
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_lags_dnn_model(data, fields, turns=600):
    """ train dnn_model with number of epochs = turns """
    if data is None or not fields:
        raise ValueError("Empty input arguments")
    np.random.seed(212)
    lags = fields
    model = get_two_layers_nnet()
    num = len(data["sunspots"].values)
    nums = np.array(np.arange(num))
    np.random.shuffle(nums)
    rands = nums[72:].tolist()
    x_train = data[lags].values[rands]
    y_train = data["sunspots"].values[rands]
    x_test = data[lags].values[nums[:72]]
    y_test = data["sunspots"].values[nums[:72]]
    mean = data[lags].values.mean(axis=0)
    std = data[lags].values.std(axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    model.fit(x_train, y_train, epochs=turns, batch_size=64,
              validation_data=(x_test, y_test))
    return model


def calculate_dnn_prediction(x_data: np.array, y_data: np.array) -> tuple:
    """ calculate dnn prediction """
    if x_data is None or y_data is None:
        raise ValueError("Empty input arguments")
    mdl = get_two_layers_nnet() if LAGS_DNN_MODEL is None else LAGS_DNN_MODEL
    means = x_data.mean()
    std = x_data.std()
    x_data = (x_data - means) / std
    result = mdl.predict(x_data).reshape(-1)
    mae = mean_absolute_error(y_data, result)
    return result, mae
