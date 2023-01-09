from flask import Blueprint, render_template
from webapp.dl_logic import create_line_plot


blueprint = Blueprint("stat", __name__, url_prefix="/stat")


@blueprint.route("/show")
def hello_word():
    return ""
