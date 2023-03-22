"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    chart views
"""
import logging
import numpy as np
from flask import Blueprint, render_template
from webapp.utils.gaps_util import fill_gaps
from webapp.utils.dataframe_util import sunspot_numbers


blueprint = Blueprint("gaps", __name__, url_prefix="/gaps")


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


@blueprint.route("/show_gaps")
def show_gaps():
    """ show_gaps function """
    data = fill_gaps()
    return render_template("chart/chart.html",
                           time=data["date"].values.tolist(),
                           y1=data["with_gap"].values.tolist(),
                           y2=data["composite"].values.tolist())


@blueprint.route("/bar_plot")
def bar_plot():
    """ bar_plot function """
    dat1, dat2 = load_sunspots_lists()
    return render_template("chart/barplot.html",
                           time=dat1[-200:],
                           y=dat2[-200:])
