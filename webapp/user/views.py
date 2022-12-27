"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    blueprint for user
"""
from flask import render_template, flash, redirect, url_for, Blueprint
from flask_login import login_user, logout_user, current_user
from webapp.user.forms import LoginForm
from webapp.business_logic import get_user_by_name

blueprint = Blueprint("user", __name__, url_prefix="/user")


@blueprint.route("/logout")
def logout():
    """ logout endpoint """
    logout_user()
    return redirect(url_for("index"))


@blueprint.route("/login")
def login():
    """ login endpoint """
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    title = "Login"
    form = LoginForm()
    return render_template("user/login.html", title=title, form=form)


@blueprint.route("/process_login", methods=["POST"])
def process_login():
    """ process login endpoint """
    form = LoginForm()
    if form.validate_on_submit():
        user = get_user_by_name(form.username.data)
        if user and user.check_password(form.password.data):
            login_user(user)
            flash("Вы вошли на сайт")
            return redirect(url_for("index"))
    flash("Неправильное имя пользователя или пароль")
    return redirect(url_for("user.login"))
