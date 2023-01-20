from flask import Blueprint, render_template
from numpy import hstack
from webapp.dl_logic import create_line_plot, predict_next_cycle, \
    predict_two_cycles


blueprint = Blueprint("chart", __name__, url_prefix="/chart")


@blueprint.route("/chart")
def draw():
    dat1, dat2 = create_line_plot()
    return render_template("chart/chart.html", x=dat1.tolist(), y=dat2.tolist())


@blueprint.route("/bar_plot")
def bar_plot():
    dat1, dat2 = create_line_plot()
    return render_template("chart/barplot.html", time=dat1[-200:].tolist(), y=dat2[-200:].tolist())


@blueprint.route("/next_cycle")
def draw_next_cycle():
    """ draw next cycle """
    years, sn = create_line_plot()
    data, times = predict_next_cycle(sn[-1152:], years[-1152:])
    dat = hstack([sn[-1152:], data])
    time = hstack((years[-1152:], times))
    return render_template("chart/two_charts.html",
                           x=time.tolist(),
                           y=dat.tolist(),
                           x2=times.tolist(),
                           y2=data.tolist())


@blueprint.route("/two_cycles")
def draw_next_two_cycles():
    """ draw next two cycles """
    years, sn = create_line_plot()
    data, times = predict_two_cycles(sn[-1152:], years[-1152:])
    return render_template("chart/predict_cycles.html",
                           x=times[-256:].tolist(),
                           y=data.tolist(),
                           x2=times[:128].tolist(),
                           y2=data[:128].tolist())
