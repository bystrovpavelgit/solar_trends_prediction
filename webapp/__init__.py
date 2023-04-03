"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    init py
"""
from flask import Flask, render_template
from flask_login import LoginManager
from webapp.db import DB
from webapp.user.models import User
from webapp.stat.views import blueprint as stat_blueprint
from webapp.user.views import blueprint as user_blueprint
from webapp.chart.views import blueprint as chart_blueprint
from webapp.gaps.views import blueprint as gaps_blueprint
from webapp.modules.error_handler import ErrorHandler
from webapp.utils.plot_util import prepare_autocorr_data, \
    plot_lags_correlation_heatmap


def create_app():
    """ create app """
    app = Flask(__name__, static_url_path="/webapp/static")
    app.config.from_pyfile("config.py")
    login_mgr = LoginManager()
    login_mgr.init_app(app)
    login_mgr.login_view = "user.login"
    DB.init_app(app)
    app.register_blueprint(stat_blueprint)
    app.register_blueprint(user_blueprint)
    app.register_blueprint(chart_blueprint)
    app.register_blueprint(gaps_blueprint)
    handler = ErrorHandler()
    handler.init_app(app)

    @login_mgr.user_loader
    def load_user(user_id):
        user = User.query.get(user_id)
        return user

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/next_menu")
    def next_menu():
        return render_template("new_menu.html")

    @app.route("/third_menu")
    def third_menu():
        return render_template("third_menu.html")

    @app.route("/autocorrelation")
    def auto_correlation():
        """ auto correlation graph """
        filename, p_val = prepare_autocorr_data()
        return render_template("gaps/show_autocorrelation.html",
                               main_img=filename,
                               p_value=p_val)

    @app.route("/heatmap")
    def lags_heatmap():
        """ show lags heatmap """
        filename = plot_lags_correlation_heatmap()
        return render_template("gaps/show_heatmap.html",
                               main_img=filename)

    return app
