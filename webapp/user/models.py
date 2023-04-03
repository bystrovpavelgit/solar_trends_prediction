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


class SunspotNumbers(DB.Model):
    """ Sunspot Numbers table"""
    __tablename__ = "sunspot_numbers"
    id = DB.Column(DB.Integer, primary_key=True)
    year_float = DB.Column(DB.Float, index=True, unique=True)
    sunspots = DB.Column(DB.Float)
    observations = DB.Column(DB.Integer)
    mean_1y = DB.Column(DB.Float)
    mean_3y = DB.Column(DB.Float)
    mean_12y = DB.Column(DB.Float)
    sunspots_max = DB.Column(DB.Float)
    sunspots_min = DB.Column(DB.Float)
    sunspots_avg = DB.Column(DB.Float)

    def __repr__(self):
        """repr method"""
        return f"<SunspotNumbers {self.year_float} {self.sunspot_num}>"
