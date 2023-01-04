from flask import Blueprint, render_template
from numpy import hstack
from webapp.dl_logic import create_line_plot, predict_next_cycle, \
    predict_two_cycles


blueprint = Blueprint("chart", __name__, url_prefix="/chart")


@blueprint.route("/chart")
def draw():
    dat1, dat2 = create_line_plot()
    return render_template("chart/chart.html", x=dat1.tolist(), y=dat2.tolist())


@blueprint.route("/next_cycle")
def draw_next_cycle():
    """ draw next cycle """
    years, sn = create_line_plot()
    data, times = predict_next_cycle(sn[-1152:], years[-1152:])
    dat = hstack([sn[-1152:], data])
    return render_template("chart/chart.html", x=list(range(1280)), y=dat.tolist())


@blueprint.route("/two_cycles")
def draw_next_two_cycles():
    """ draw next two cycles """
    years, sn = create_line_plot()
    data, times = predict_two_cycles(sn[-1152:], years[-1152:])
    return render_template("chart/two_charts.html",
                           x=list(range(256)),
                           y=data.tolist(),
                           x2=list(range(128)),
                           y2=data[:128].tolist())
