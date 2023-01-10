import csv
from sqlite3 import IntegrityError
from sqlalchemy.exc import SQLAlchemyError, PendingRollbackError
from webapp import create_app
from webapp.db import DB
from webapp.stat.models import SunspotNumbers
from webapp.utils.enrich_sunspots import get_enriched_dataframe


def insert_authors(data):
    """ insert multiple authors """
    if data:
        try:
            DB.session.bulk_insert_mappings(SunspotNumbers, data)
            DB.session.commit()
        except (SQLAlchemyError, IntegrityError, PendingRollbackError) as e:
            error = str(e.__dict__['orig'])
            err = f"Exception in insert_authors: {error} "
            print(err)
            DB.session.rollback()


def process_authors_from_file(name="data/authors.csv"):
    """ process authors from file """
    columns = ["year_float",
               "sunspots",
               "observations",
               "mean_1y",
               "mean_3y",
               "mean_12y",
               "sunspots_max",
               "sunspots_min",
               "sunspots_avg"]
    res = []
    try:
        with open(name, "r") as f:
            fields = columns
            rows = csv.DictReader(f, fields, delimiter=',')
            res = [{"id": int(row["user_id"]), "name": row["name"]} for row in
                   rows if row["user_id"] != "user_id"]
            insert_authors(res)
    except FileNotFoundError:
        print(f"File not found: {name}")
    return res


if __name__ == "__main__":
    app = create_app()
    with app.app_context():
        df = get_enriched_dataframe(csf_file="data/solarn_month.csv")
        # load all authors
        result = process_authors_from_file(name="data/authors.csv")
        print(f"{len(result)} authors loaded")
