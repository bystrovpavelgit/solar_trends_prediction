"""
    Apache License 2.0 Copyright (c) 2023 Pavel Bystrov
    error handler
"""
from typing import Callable
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.exceptions import HTTPException
from webapp.utils.default.error_handler import exception_handler, \
    handler_http_exception, handler_input_value_exception, \
    handler_file_not_found_exception, handler_sql_exception


class ErrorHandler:
    """ Error Handler """
    _handlers = {
        SQLAlchemyError: handler_sql_exception,
        FileNotFoundError: handler_file_not_found_exception,
        ValueError: handler_input_value_exception,
        HTTPException: handler_http_exception,
        Exception: exception_handler}

    def __init__(self):
        self._app = None

    def init_app(self, app):
        """ init app """
        self._app = app

    @classmethod
    def add_error_handler(cls, exception: Exception, handler: Callable):
        """ add error handler """
        cls._handlers[exception] = handler

    def _register_errors_handlers(self):
        for exception, func in self._handlers.items():
            self._app.register_error_handler(exception, func)
