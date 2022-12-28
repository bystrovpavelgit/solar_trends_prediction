"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
   trends sql model
"""
from webapp.db import DB


class SunspotNumbers(DB.Model):
    """ Sunspot Numbers table"""
    __tablename__ = "sunspot_numbers"
    id = DB.Column(DB.Integer, primary_key=True)
    date = DB.Column(DB.Datetime, index=True, unique=True)
    sunspot_num = DB.Column(DB.Float)

    def __repr__(self):
        """repr method"""
        return f"<SunspotNumber {self.date} {self.sunspot_num}>"


class Measure(DB.Model):
    """ table with Measures """
    __tablename__ = "measure"
    id = DB.Column(DB.Integer, primary_key=True)
    date = DB.Column(DB.Datetime, index=True, unique=True)
    sunspot_num = DB.Column(DB.Float)

    def __repr__(self):
        """repr method"""
        return f"<Measure {self.id} {self.sunspot_num}>"
