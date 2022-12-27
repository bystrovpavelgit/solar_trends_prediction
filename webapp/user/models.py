"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    models for SQLite
"""
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from webapp.db import DB


class User(DB.Model, UserMixin):
    """User model"""
    id = DB.Column(DB.Integer, primary_key=True)
    username = DB.Column(DB.String(80), index=True, unique=True)
    password = DB.Column(DB.String(128))
    role = DB.Column(DB.String(15), index=True)

    @property
    def is_admin(self):
        """is admin"""
        return self.role == "admin"

    def __repr__(self):
        """repr method"""
        return "<User {} {}>".format(self.id, self.username)

    def set_password(self, password):
        """set password"""
        self.password = generate_password_hash(password)

    def check_password(self, password):
        """check password"""
        return check_password_hash(self.password, password)
