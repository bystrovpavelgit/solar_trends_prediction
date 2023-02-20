""" chart views """
from flask import Blueprint, render_template
from numpy import hstack
from webapp.dl_logic import sunspot_numbers, predict_next_cycle, \
    predict_two_cycles
from webapp.config import RNN_INPUT_SIZE, RNN_OUTPUT_SIZE

blueprint = Blueprint("chart", __name__, url_prefix="/chart")


@blueprint.route("/chart")
def draw():
    """ draw function """
    dat1, dat2 = sunspot_numbers()
    return render_template("chart/chart.html",
                           x=dat1.tolist(),
                           y=dat2.tolist())


@blueprint.route("/bar_plot")
def bar_plot():
    """ bar_plot function """
    dat1, dat2 = sunspot_numbers()
    return render_template("chart/barplot.html",
                           time=dat1[-200:].tolist(),
                           y=dat2[-200:].tolist())


@blueprint.route("/next_cycle")
def draw_next_cycle():
    """ draw next cycle """
    years, spots = sunspot_numbers()
    data, times = predict_next_cycle(spots[-RNN_INPUT_SIZE:],
                                     years[-RNN_INPUT_SIZE:])
    dat = hstack([spots[-RNN_INPUT_SIZE:], data])
    time = hstack((years[-RNN_INPUT_SIZE:], times))
    return render_template("chart/two_charts.html",
                           x=time.tolist(),
                           y=dat.tolist(),
                           x2=times.tolist(),
                           y2=data.tolist())


@blueprint.route("/two_cycles")
def draw_next_two_cycles():
    """ draw next two cycles """
    double_size = 2 * RNN_OUTPUT_SIZE
    years, spots = sunspot_numbers()
    data, times = predict_two_cycles(spots[-RNN_INPUT_SIZE:],
                                     years[-RNN_INPUT_SIZE:])
    return render_template("chart/predict_cycles.html",
                           x=times[-double_size:].tolist(),
                           y=data.tolist(),
                           x2=times[:RNN_OUTPUT_SIZE].tolist(),
                           y2=data[:RNN_OUTPUT_SIZE].tolist())
