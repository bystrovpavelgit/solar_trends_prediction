""" Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    initialize database """
from webapp.db import DB
from webapp import create_app

if __name__ == "__main__":
    app = create_app()
    DB.create_all(app=app)
