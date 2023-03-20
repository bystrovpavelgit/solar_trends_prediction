"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    statistical views
"""
import logging
from flask import Blueprint, render_template, request, flash, redirect
from webapp.config import VALID_VALUES, REGRESSION_VALUES
from webapp.utils.dataframe_util import get_enriched_dataframe, prepare_data
from webapp.utils.enrich_sunspots import get_results_for_best_classifier
from webapp.utils.trends_util import get_fourier_prediction, \
    prediction_by_type
from webapp.stat.api import get_smoothed_data_by_type

blueprint = Blueprint("stat", __name__, url_prefix="/stat")


def log_and_flash(msg: str) -> None:
    """ logging """
    logging.warning(msg)
    flash(msg)


@blueprint.route("/smoothing_curve", methods=["GET", "POST"])
def process_smoothing():
    """ show smoothed curve according to type """
    selected = VALID_VALUES[0]
    if request.method == "POST":
        type_ = request.form.get("smoothing")
        selected = type_
        if type_ is None or type_ not in VALID_VALUES:
            log_and_flash(f"неверный тип сглаживания: {type_}")
            return redirect("/")
    result = get_smoothed_data_by_type(selected)
    return render_template("stat/select_graph.html",
                           title="Выбор сглаживания",
                           selected=selected,
                           time=result[0],
                           y=result[1],
                           y2=result[2])


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


@blueprint.route("/regression", methods=["GET", "POST"])
def regression_prediction():
    """ display linear regression predictions """
    selected = REGRESSION_VALUES[0]
    if request.method == "POST":
        type_ = request.form.get("regression")
        selected = type_
        if type_ not in REGRESSION_VALUES:
            log_and_flash(f"неверный тип регрессии: {type_}")
            return redirect("/")
    data = prepare_data()
    time = data["year_float"].values.tolist()
    sunspots = data["sunspots"].values.tolist()
    predicted, mae = prediction_by_type(selected, data)
    print(f"MAE: {mae}")
    return render_template("stat/select_regression.html",
                           title="Тип регрессии",
                           selected=selected,
                           time=time,
                           y=sunspots,
                           y2=predicted.tolist())
