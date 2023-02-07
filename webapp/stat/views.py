""" statistical views """
from flask import Blueprint, render_template
from webapp.utils.enrich_sunspots import get_enriched_dataframe, \
    get_results_for_best_classifier
from webapp.utils.trends_util import exponential_smoothing, \
    double_exponential_smoothing, hw_exponential_smoothing, \
    get_fourier_prediction

blueprint = Blueprint("stat", __name__, url_prefix="/stat")


@blueprint.route("/rolling_means")
def rolling_means():
    """ display rolling means """
    data = get_enriched_dataframe()
    info = {'graph': 'Moving average for sunspots numbers'}
    time = data["year_float"].values.tolist()
    period = len(time)
    sunspots = data["sunspots"].values.tolist()
    mean3 = data["mean_3y"].values.tolist()
    mean12 = data["mean_12y"].values.tolist()
    return render_template("stat/charts.html",
                           info=info,
                           time=time,
                           y=sunspots[:period],
                           y2=mean3[:period],
                           y3=mean12[:period])


@blueprint.route("/exp_smoothing")
def exp_smoothing():
    """ display exponential smoothing """
    data = get_enriched_dataframe()
    time = data["year_float"].values.tolist()
    period = 1200  # show 100 years
    sunspots = data["sunspots"].values.tolist()
    exp = exponential_smoothing(data["sunspots"], .25)
    duo_exp = double_exponential_smoothing(data["sunspots"], .2, .2)
    return render_template("stat/exp_smoothing.html",
                           time=time[-period:],
                           y=sunspots[-period:],
                           y2=exp[-period:],
                           y3=duo_exp[-period:])


@blueprint.route("/holt_winters")
def holt_winters():
    """ display Holt-winters exponential smoothing """
    data = get_enriched_dataframe()
    time = data["year_float"].values.tolist()
    sunspots = data["sunspots"].values.tolist()
    triple = hw_exponential_smoothing(data.sunspots)
    return render_template("stat/triple_smoothing.html",
                           time=time,
                           y=sunspots,
                           y2=triple)


@blueprint.route("/best")
def best_model():
    """ display results for best ML model """
    info = {'graph': 'Adaboost classifier predictions for max and min'}
    time, pmax, pmin, max_, sunspots = get_results_for_best_classifier()
    period = len(time)
    timeseries = time[:period].tolist()
    return render_template("stat/best.html",
                           info=info,
                           time=timeseries,
                           y=(pmax[:period] * 50).tolist(),
                           y2=(pmin[:period] * 50).tolist(),
                           y3=max_[:period].tolist(),
                           y4=sunspots[:period].tolist())


@blueprint.route("/fourier")
def fourier():
    """ display fourier method predictions """
    data = get_enriched_dataframe()
    time = data["year_float"].values
    sunspots = data["sunspots"].values
    preds, time2 = get_fourier_prediction(sunspots, time, 300)
    return render_template("stat/fourier.html",
                           time=time.tolist(),
                           y=sunspots.tolist(),
                           time2=time2.tolist(),
                           y2=preds.tolist())
