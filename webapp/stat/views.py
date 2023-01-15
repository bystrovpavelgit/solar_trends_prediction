from flask import Blueprint, render_template
from webapp.utils.enrich_sunspots import get_enriched_dataframe, \
    get_results_for_best_classifier

blueprint = Blueprint("stat", __name__, url_prefix="/stat")


@blueprint.route("/rolling_means")
def rolling_mean():
    df = get_enriched_dataframe()
    info = {'graph': 'Moving average for sunspots numbers'}
    time = df["year_float"].values.tolist()
    period = len(time)
    y1 = df["sunspots"].values.tolist()
    y2 = df["mean_3y"].values.tolist()
    y3 = df["mean_12y"].values.tolist()
    return render_template("stat/charts.html",
                           info=info,
                           time=time[:period],
                           y=y1[:period],
                           y2=y2[:period],
                           y3=y3[:period])


@blueprint.route("/best")
def best_model():
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
