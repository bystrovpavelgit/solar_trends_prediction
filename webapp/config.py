"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    web-app configuration
"""
import logging
import os
from datetime import timedelta

logging.basicConfig(filename='webapp.log', level=logging.DEBUG)
basedir = os.path.abspath(os.path.dirname(__file__))
SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(basedir,
                                                      "..",
                                                      "webapp.db")
SECRET_KEY = "ASDWQYUhj342678gvmjhxckbdvkjbscde"
REMEMBER_COOKIE_DURATION = timedelta(days=7)
SQLALCHEMY_TRACK_MODIFICATIONS = True
