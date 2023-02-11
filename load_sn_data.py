"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    sunspots data loader for database
"""
import csv
from sqlite3 import IntegrityError
from sqlalchemy.exc import SQLAlchemyError, PendingRollbackError
from webapp import create_app
from webapp.db import DB
from webapp.user.models import SunspotNumbers
from webapp.utils.enrich_sunspots import get_enriched_dataframe

columns_ = ["year_float",
            "sunspots",
            "observations",
            "mean_1y",
            "mean_3y",
            "mean_12y",
            "sunspots_max",
            "sunspots_min",
            "sunspots_avg"]


def create_mapping(rows):
    """ create mapping """
    result = [{"year_float": float(row["year_float"]),
               "sunspots": row["sunspots"],
               "observations": int(row["observations"]),
               "mean_1y": row["mean_1y"],
               "mean_3y": row["mean_3y"],
               "mean_12y": row["mean_12y"],
               "sunspots_max": row["sunspots_max"],
               "sunspots_min": row["sunspots_min"],
               "sunspots_avg": row["sunspots_avg"],
               } for row in rows
              if row["year_float"] != "year_float"]
    return result


def insert_sunspots(data):
    """ insert multiple authors """
    if data:
        try:
            DB.session.bulk_insert_mappings(SunspotNumbers, data)
            DB.session.commit()
        except (SQLAlchemyError, IntegrityError, PendingRollbackError) as ex:
            error = str(ex.__dict__['orig'])
            err = f"Exception in insert_authors: {error} "
            print(err)
            DB.session.rollback()


def process_data_from_file(name="data/sunspot_numbers_enriched.csv"):
    """ process authors from file """
    columns = ["Year",
               "month",
               "year_float",
               "sunspots",
               "std",
               "observations",
               "def_or_prov",
               "mean_1y",
               "mean_3y",
               "mean_12y",
               "sunspots_max",
               "sunspots_min",
               "sunspots_avg"]
    mapping = []
    try:
        with open(name, "r", encoding="utf-8") as fil:
            rows = csv.DictReader(fil, columns, delimiter=';')
            mapping = create_mapping(rows)
            insert_sunspots(mapping)
    except FileNotFoundError:
        print(f"File not found: {name}")
    return mapping


if __name__ == "__main__":
    app = create_app()
    df = get_enriched_dataframe()
    df.to_csv("data/sunspot_numbers_enriched.csv", sep=";", index=False)
    with app.app_context():
        res = process_data_from_file(name="data/sunspot_numbers_enriched.csv")
        print(f"{len(res)} sunspot numbers loaded")
