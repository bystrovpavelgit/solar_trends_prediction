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
