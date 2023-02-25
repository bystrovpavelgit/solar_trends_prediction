"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    chart views
"""
import logging
import numpy as np
from flask import Blueprint, render_template
from numpy import hstack
from webapp.utils.dataframe_util import sunspot_numbers
from webapp.dl_logic import predict_next_cycle, predict_two_cycles
from webapp.config import RNN_INPUT_SIZE, RNN_OUTPUT_SIZE

blueprint = Blueprint("chart", __name__, url_prefix="/chart")


def load_sunspots_arrays():
    """ load sunspots arrays and log error if empty"""
    opt_res = sunspot_numbers()
    if opt_res.is_empty():
        logging.error("File not found in load_sunspots_arrays func")
        res1 = res2 = np.array([0])
    else:
        res1, res2 = opt_res.get()
    return res1, res2


def load_sunspots_lists():
    """ load sunspot numbers data and time intervals as lists """
    data1, data2 = load_sunspots_arrays()
    data1 = data1.tolist()
    data2 = data2.tolist()
    return data1, data2


@blueprint.route("/chart")
def draw():
    """ draw function """
    dat1, dat2 = load_sunspots_lists()
    return render_template("chart/chart.html", x=dat1, y=dat2)


@blueprint.route("/bar_plot")
def bar_plot():
    """ bar_plot function """
    dat1, dat2 = load_sunspots_lists()
    return render_template("chart/barplot.html", time=dat1[-200:], y=dat2[-200:])


@blueprint.route("/next_cycle")
def draw_next_cycle():
    """ draw next cycle """
    years, spots = load_sunspots_arrays()
    if len(years) > 1:
        data, times = predict_next_cycle(spots[-RNN_INPUT_SIZE:],
                                         years[-RNN_INPUT_SIZE:])
        predicted = data.tolist()
        times = times.tolist()
        dat = hstack([spots[-RNN_INPUT_SIZE:], data]).tolist()
        time = hstack((years[-RNN_INPUT_SIZE:], times)).tolist()
    else:
        times = time = years.tolist()
        dat = predicted = spots.tolist()
    return render_template("chart/two_charts.html",
                           x=time,
                           y=dat,
                           x2=times,
                           y2=predicted)


@blueprint.route("/two_cycles")
def draw_next_two_cycles():
    """ draw next two cycles """
    double_size = 2 * RNN_OUTPUT_SIZE
    years, spots = load_sunspots_arrays()
    if len(years) > 1:
        data, times = predict_two_cycles(spots[-RNN_INPUT_SIZE:],
                                         years[-RNN_INPUT_SIZE:])
        data = data.tolist()
        times = times.tolist()
    else:
        times = data = [0]
    return render_template("chart/predict_cycles.html",
                           x=times[-double_size:],
                           y=data,
                           x2=times[RNN_OUTPUT_SIZE: double_size],
                           y2=data[RNN_OUTPUT_SIZE: double_size])
