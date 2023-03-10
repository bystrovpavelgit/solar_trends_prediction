"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    chart views
"""
import logging
import numpy as np
from flask import Blueprint, render_template
from numpy import hstack
from webapp.config import RNN_INPUT_SIZE, RNN_OUTPUT_SIZE
from webapp.dl_logic import predict_next_cycle, predict_two_cycles
from webapp.utils.dataframe_util import sunspot_numbers, \
    get_enriched_dataframe


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


def count_sunspots_data(dframe):
    """ count sunspots data """
    cent1 = dframe[dframe["year_float"] < 1800.]["year_float"].count()
    cond1 = (dframe["year_float"] < 1900.) & (1800. <= dframe["year_float"])
    cent2 = dframe[cond1]["year_float"].count()
    cond2 = (dframe["year_float"] < 2000.) & (1900. <= dframe["year_float"])
    cent3 = dframe[cond2]["year_float"].count()
    cent4 = dframe[dframe["year_float"] >= 2000.]["year_float"].count()
    result = [cent1, cent2, cent3, cent4]
    return result


@blueprint.route("/chart")
def draw():
    """ draw function """
    dat1, dat2 = load_sunspots_lists()
    return render_template("chart/chart.html", x=dat1, y=dat2)


@blueprint.route("/input_stat")
def show_data():
    """ show data availability using pie-chart """
    data = get_enriched_dataframe()
    result = count_sunspots_data(data)
    labels = [18, 19, 20, 21]
    return render_template("chart/input_statistics.html", data=labels, y=result)


@blueprint.route("/bar_plot")
def bar_plot():
    """ bar_plot function """
    dat1, dat2 = load_sunspots_lists()
    return render_template("chart/barplot.html",
                           time=dat1[-200:],
                           y=dat2[-200:])


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
