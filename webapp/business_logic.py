"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    business logic for trends prediction app
"""
import logging
from sqlite3 import IntegrityError
from sqlalchemy.exc import SQLAlchemyError
from webapp.user.models import User


def get_user_by_name(name):
    """  get user by name """
    user = None
    try:
        user = User.query.filter_by(username=name).first()
    except (SQLAlchemyError, IntegrityError) as ex:
        error = str(ex.__dict__['orig'])
        msg = f"Exception in get_user_by_name: {error} / {name}"
        logging.error(msg)
    return user
