"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    web-app configuration
"""
import logging
import os
from datetime import timedelta

logging.basicConfig(filename='webapp.log', level=logging.INFO)
basedir = os.path.abspath(os.path.dirname(__file__))
SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(basedir,
                                                      "..",
                                                      "webapp.db")
SECRET_KEY = "QWEWQYUhj342678gvmjhxckbdvkjbs321"
REMEMBER_COOKIE_DURATION = timedelta(days=7)
