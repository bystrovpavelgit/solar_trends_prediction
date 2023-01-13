from flask import Blueprint, render_template
from webapp.utils.enrich_sunspots import get_enriched_dataframe, \
    get_results_for_best_classifier

blueprint = Blueprint("stat", __name__, url_prefix="/stat")


@blueprint.route("/rolling_means")
def rolling_mean():
    df = get_enriched_dataframe()
    time = df["year_float"].values.tolist()
    y1 = df["sunspots"].values.tolist()
    y2 = df["mean_3y"].values.tolist()
    return render_template("stat/two_charts.html",
                           x=time[:128],
                           y=y1[:128],
                           x2=time[:128],
                           y2=y2[:128])


@blueprint.route("/best")
def best_model():
    time, pmax, pmin, max_, sunspots = get_results_for_best_classifier()
    timeseries = time[:128].tolist()
    return render_template("stat/best.html",
                           time=timeseries,
                           y=pmax[:128].tolist(),
                           y2=pmin[:128].tolist(),
                           y3=max_[:128].tolist(),
                           y4=sunspots[:128].tolist())
