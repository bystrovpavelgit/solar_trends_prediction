"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    init py
"""
from flask import Flask, render_template
from flask_login import LoginManager
from webapp.business_logic import create_line_plot
from webapp.db import DB
from webapp.user.models import User
from webapp.user.views import blueprint as user_blueprint


def create_app():
    """ create app """
    app = Flask(__name__, static_url_path="/", static_folder="/")
    app.config.from_pyfile("config.py")
    login_mgr = LoginManager()
    login_mgr.init_app(app)
    login_mgr.login_view = "user.login"
    DB.init_app(app)
    app.register_blueprint(user_blueprint)

    @login_mgr.user_loader
    def load_user(user_id):
        user = User.query.get(user_id)
        return user

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/chart")
    def draw():
        dat1, dat2 = create_line_plot()
        return render_template("chart.html", x=dat1, y=dat2)

    return app
