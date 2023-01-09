"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
   trends sql model
"""
from webapp.db import DB


class SunspotNumbers(DB.Model):
    """ Sunspot Numbers table"""
    __tablename__ = "sunspot_numbers"
    id = DB.Column(DB.Integer, primary_key=True)
    year_float = DB.Column(DB.Float, index=True, unique=True)
    sunspot_num = DB.Column(DB.Float)
    measurements = DB.Column(DB.Integer)
    mean_1y = DB.Column(DB.Float)
    mean_2y = DB.Column(DB.Float)
    mean_2y = DB.Column(DB.Float)
    spotnum_max = DB.Column(DB.Float)
    spotnum_min = DB.Column(DB.Float)
    spotnum_avg = DB.Column(DB.Float)

    def __repr__(self):
        """repr method"""
        return f"<SunspotNumbers {self.year_float} {self.sunspot_num}>"


class Measure(DB.Model):
    """ table with Measures """
    __tablename__ = "measure"
    id = DB.Column(DB.Integer, primary_key=True)
    year = DB.Column(DB.Integer, index=True, unique=False)
    year_float = DB.Column(DB.Float, index=True, unique=True)
    sunspot_num = DB.Column(DB.Float)

    def __repr__(self):
        """repr method"""
        return f"<Measure {self.id} {self.sunspot_num}>"
