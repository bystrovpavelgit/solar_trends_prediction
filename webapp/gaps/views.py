"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    views to display gaps in the data
"""
from flask import Blueprint, render_template
from webapp.gaps.api import count_gaps
from webapp.utils.gaps_util import fill_gaps


blueprint = Blueprint("gaps", __name__, url_prefix="/gaps")


@blueprint.route("/fill_gaps")
def all_gaps():
    """ fill_gaps function """
    data = fill_gaps()
    return render_template("gaps/show_gaps.html",
                           time=data["date"].values[:1200].tolist(),
                           y=data["with_gap"].values[:1200].tolist(),
                           y2=data["composite"].values[:1200].tolist())


@blueprint.route("/gaps_stat")
def gaps_stat():
    """ gaps statistics """
    data = fill_gaps()
    total = len(data["with_gap"])
    gaps_num = count_gaps(data["with_gap"])
    arr = [(total - gaps_num), gaps_num]
    return render_template("gaps/input_statistics.html", y=arr)
