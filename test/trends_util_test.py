""" Trends Util unit-tests """
import unittest
import numpy as np
import pandas as pd
from webapp.utils.enrich_sunspots import get_enriched_dataframe
from webapp.utils.trends_util import moving_average, rolling_mean, min_index,\
    exponential_smoothing, double_exponential_smoothing, find_minimums, \
    get_optimal_params, hw_exponential_smoothing


class TrendsUtilTest(unittest.TestCase):
    """ Trends Util test """
    def setUp(self):
        """ Инит TestUser """
        self.max_len = 5

    def test_moving_average(self):
        """ moving_average юнит-тест """
        data = [4, 6, 4, 6, 4, 6]

        avg = moving_average(data, 6)

        self.assertEqual(avg, 5, "равен 5")

    def test_sum(self):
        """ summation юнит-тест """
        self.assertEqual(sum([3, 2]), 5, "равен 5")

    def test_rolling_mean(self):
        """ rolling_mean юнит-тест """
        date = pd.Series([0, 4, 6, 4, 6, 4, 6])

        stat = rolling_mean(date, 6).values.tolist()

        self.assertIsNotNone(stat)
        self.assertEqual(stat[5], 4., "равен 4")
        self.assertEqual(stat[6], 5., "равен 5")

    def test_exponential_smoothing(self):
        """ exponential_smoothing юнит-тест """
        date = [4, 6, 4, 6, 4, 6]

        res = exponential_smoothing(date, 0.5)

        self.assertIsNotNone(res)
        self.assertEqual(res[1:5], [5.0, 4.5, 5.25, 4.625],
                         "равен [5.0, 4.5, 5.25, 4.625]")

    def test_double_exponential_smoothing(self):
        """ double_exponential_smoothing юнит-тест """
        date = [4, 6, 4, 6, 4, 6]

        res = double_exponential_smoothing(date, 0.5, 0.5)

        self.assertIsNotNone(res)
        self.assertEqual(res[1:5], [8.0, 7.0, 7.25, 5.5625],
                         "равен (8.0, 7.0, 7.25, 5.5625)")

    def test_min_index_positively(self):
        """ позитивный юнит-тест для min_index """
        arr = np.array([2, 3, 5, 5, 3, 1, 2])

        index = min_index(arr, 0, 6)

        self.assertEqual(index, 5, "min_index = 5")

    def test_min_index_negatively(self):
        """ негативный юнит-тест для min_index """
        empty = np.array([])

        index = min_index(empty, 0, 9)

        self.assertEqual(index, -1, "min_index равен -1")

    def test_find_minimums_negatively(self):
        """ негативный юнит-тест для min_index """
        empty = np.array([])

        arr = find_minimums(empty, 9)

        self.assertEqual(arr, [], "minimums = []")

    def test_find_minimums_positively(self):
        """ негативный юнит-тест для min_index """
        inp = np.array([2, 3, 5, 5, 3, 1, 2])

        arr = find_minimums(inp, 7)

        self.assertEqual(arr, [5], "minimums равен [5]")

    def test_get_optimal_params(self):
        """ юнит-тест для get_optimal_params func """
        dataset = get_enriched_dataframe()

        alpha, beta, gamma = get_optimal_params(dataset.sunspots)

        self.assertTrue(alpha > 0., "alpha > 0")
        self.assertTrue(beta >= 0., "beta >= 0")
        self.assertTrue(gamma > 0., "gamma > 0.")

    def test_hw_exponential_smoothing(self):
        """ юнит-тест для triple exponential smoothing """
        data = get_enriched_dataframe()
        size = len(data["sunspots"].values)

        result = hw_exponential_smoothing(data.sunspots)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), size, f"minimums равен {size}")
        self.assertTrue(result[-1] > 0, " > 0")
