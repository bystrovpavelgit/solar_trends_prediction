""" unit test """
import pandas as pd
import unittest
from webapp import create_app
from webapp.business_logic import get_user_by_name
from webapp.dl_logic import sunspot_numbers


class TestUser(unittest.TestCase):
    """ test user """

    def setUp(self):
        """ Инит TestUser """
        self.max_len = 500

    def test_sum(self):
        """ summation юнит-тест """
        self.assertEqual(sum([3, 2]), 5, "равен 5")

    def test_get_user_by_name(self):
        """ test get_user_by_name """
        app = create_app()
        with app.app_context():
            user = get_user_by_name("gav")

        self.assertIsNone(user, "user not None")

    def test_create_line_plot(self):
        """ test create_line_plot """
        data = pd.read_csv("data/sunspot_numbers.csv", delimiter=";")
        year = data['year_float'].values.tolist()
        spots = data['sunspots'].values.tolist()

        res1, res2 = sunspot_numbers()
        res1, res2 = list(res1), list(res2)

        self.assertIsNotNone(res1, "year not None")
        self.assertIsNotNone(res2, "sunspots not None")
        self.assertEqual(res1, year, "res1 == years")
        self.assertEqual(res2, spots, "res2 == spots")
