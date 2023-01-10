from flask import Blueprint, render_template
from webapp.utils.enrich_sunspots import get_enriched_dataframe

blueprint = Blueprint("stat", __name__, url_prefix="/stat")


@blueprint.route("/rolling_means")
def rolling_mean():
    df = get_enriched_dataframe()
    time = df["year_float"].values.tolist()
    y1 = df["sunspots"].values.tolist()
    y2 = df["mean_3y"].values.tolist()
    return render_template("chart/two_charts.html",
                           x=time[:256],
                           y=y1[:256],
                           x2=time[:256],
                           y2=y2[:256])
