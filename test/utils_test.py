""" unit test """
import pandas as pd
import unittest
from webapp import create_app
from webapp.business_logic import get_user_by_name
from webapp.dl_logic import create_line_plot


class TestUser(unittest.TestCase):
    """ test user """

    def setUp(self):
        """ Инит TestUser """
        self.max_len = 500

    def test_sum(self):
        """ summation юнит-тест """
        print("юнит-тест sum\n")
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

        dat1, dat2 = create_line_plot()
        dat1, dat2 = list(dat1), list(dat2)

        self.assertIsNotNone(dat1, "year not None")
        self.assertIsNotNone(dat2, "sunspots not None")
        self.assertEqual(dat1, year, "user.username == gav")
        self.assertEqual(dat2, spots, "user.username == gav")
