"""
    Apache License 2.0 Copyright (c) 2023 Pavel Bystrov
    error handler functions
"""
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.exceptions import HTTPException


def exception_handler(err: Exception):
    """ common exception handler"""
    res = ({'message': type(err).__qualname__.lower(),
            'detail': str(err)}, 500, {'Content-Type': 'application/json'}
           )
    return res


def handler_http_exception(err: HTTPException):
    """ handle http_exception """
    res = ({'message': type(err).__qualname__.lower(),
            'detail': str(err)}, err.code, {'Content-Type': 'application/json'}
           )
    return res


def handler_input_value_exception(err: ValueError):
    """ handle input_value_exception """
    return exception_handler(err)


def handler_file_not_found_exception(err: FileNotFoundError):
    """ handle file_not_found_exception """
    return exception_handler(err)


def handler_sql_exception(err: SQLAlchemyError):
    """ handle sql_exception """
    return exception_handler(err)
