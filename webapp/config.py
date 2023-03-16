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
SECRET_KEY = "ASDWQYUhj342678gvmjhxckbdvkjbscde"
REMEMBER_COOKIE_DURATION = timedelta(days=15)
SQLALCHEMY_TRACK_MODIFICATIONS = True
VALID_VALUES = ["скользязщее среднее 3г", "скользязщее среднее 12л",
                "экспоненциальный", "двойной", "тройной"]
REGRESSION_VALUES = ["Linear", "Lasso", "Ridge"]
RNN_INPUT_SIZE = 1152
RNN_OUTPUT_SIZE = 128
