"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    views to display gaps in the data
"""
import numpy as np
from scipy import stats
from flask import Blueprint, render_template
from webapp.gaps.api import count_gaps
from webapp.utils.dataframe_util import get_all_minimums, \
    get_enriched_dataframe
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


@blueprint.route("/solar_cycles")
def solar_cycles():
    """ draw distribution plot with solar cycle durations """
    data = get_enriched_dataframe()
    lst = get_all_minimums(data["sunspots"].values, 128)
    cycles = [lst[i] - lst[i - 1] for i in range(1, len(lst))]
    num = len(cycles)
    cycles = np.array(sorted(cycles))
    std = cycles.std() / 12
    x_mean = cycles.mean() / 12
    print(std, x_mean)
    print("min / max:", np.min(cycles) / 12, np.max(cycles) / 12)
    # interval using normal distribution
    z_a = stats.t.ppf(0.997, num - 1)  # alpha = 0.006
    x_intv = std * z_a / np.sqrt(num)
    print(f"Student interval [{x_mean - x_intv}, {x_mean + x_intv}]")
    # interval using normal distribution
    z_a = stats.norm.ppf(0.997)  # alpha = 0.006
    x_intv = std * z_a / np.sqrt(num)
    print(f"Norm interval [{x_mean - x_intv}, {x_mean + x_intv}]")
    # min amount
    z_a = stats.norm.ppf(0.995)
    num = ((z_a * 1.4 / 1.1) ** 2)
    num = int(num + 0.9)  # top
    print(f"min N for sigma = 1.4, alpha = 0.01 and interval = 1.1 is {num}")
    return render_template("gaps/dist_plot.html", y=cycles.tolist())
