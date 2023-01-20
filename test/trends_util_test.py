""" Trends Util unit-tests """
import unittest
import pandas as pd
from webapp.utils.trends_util import moving_average, rolling_mean, \
    exponential_smoothing, double_exponential_smoothing


class TrendsUtilTest(unittest.TestCase):
    """ Trends Util test """
    def setUp(self):
        """ Инит TestUser """
        self.max_len = 500

    def test_moving_average(self):
        """ moving_average юнит-тест """
        data = [4, 6, 4, 6, 4, 6]

        avg = moving_average(data, 6)

        self.assertEqual(avg, 5, "равен 5")

    def test_min_index(self):
        """ min_index юнит-тест """
        self.assertEqual(sum([3, 2]), 5, "равен 5")

    def test_rolling_mean(self):
        """ min_index юнит-тест """
        date = pd.Series([0, 4, 6, 4, 6, 4, 6])

        stat = rolling_mean(date, 6).values.tolist()

        self.assertIsNotNone(stat)
        self.assertEqual(stat[5], 4., "равен 4")
        self.assertEqual(stat[6], 5., "равен 5")

    def test_exponential_smoothing(self):
        """ min_index юнит-тест """
        date = [4, 6, 4, 6, 4, 6]

        res = exponential_smoothing(date, 0.5)

        self.assertIsNotNone(res)
        self.assertEqual(res[1:5], [5.0, 4.5, 5.25, 4.625],
                         "равен 5.0, 4.5, 5.25")

    def test_double_exponential_smoothing(self):
        """ min_index юнит-тест """
        date = [4, 6, 4, 6, 4, 6]

        res = double_exponential_smoothing(date, 0.5, 0.5)

        self.assertIsNotNone(res)
        self.assertEqual(res[1:5], [8.0, 7.0, 7.25, 5.5625],
                         "равен 8.0, 7.0, 7.25, 5.5625")
