"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    business logic for trends prediction app
"""
import logging
import pandas as pd
from sqlite3 import IntegrityError
from sqlalchemy.exc import SQLAlchemyError
from webapp.user.models import User


def get_user_by_name(name):
    """  get user by name """
    try:
        user = User.query.filter_by(username=name).first()
        return user
    except (SQLAlchemyError, IntegrityError) as ex:
        error = str(ex.__dict__['orig'])
        logging.error(f"Exception in get_user_by_name: {error} / {name}")


def create_line_plot():
    """ create line plot """
    df = pd.read_csv("data/solarn_month.csv", delimiter=";")
    return df['year_float'].values.tolist(), df['sunspots'].values.tolist()
