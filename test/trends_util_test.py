"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    Trends Util unit-tests
"""
import unittest
import numpy as np
import pandas as pd
from webapp.utils.dataframe_util import get_enriched_dataframe
from webapp.utils.trends_util import exponential_smoothing, \
    double_exponential_smoothing, get_optimal_params, hw_exponential_smoothing


class TrendsUtilTest(unittest.TestCase):
    """ Trends Util test """

    def test_sum(self):
        """ summation юнит-тест """
        self.assertEqual(sum([3, 2]), 5, "равен 5")

    def test_exponential_smoothing(self):
        """ exponential_smoothing юнит-тест """
        date = [4, 6, 4, 6, 4, 6]

        res = exponential_smoothing(date, 0.5)

        self.assertIsNotNone(res)
        self.assertEqual(res[1:5], [5.0, 4.5, 5.25, 4.625],
                         "равен [5.0, 4.5, 5.25, 4.625]")

    def test_exponential_smoothing_negatively2(self):
        """ exponential_smoothing юнит-тест """
        try:
            res = exponential_smoothing([4], -0.5)

            self.assertIsNotNone(res)
        except ValueError as err:
            self.assertIsNotNone(err, "ValueError")

    def test_exponential_smoothing_negatively3(self):
        """ exponential_smoothing юнит-тест """
        try:
            res = exponential_smoothing([], 0.1)

            self.assertIsNotNone(res)
        except ValueError as err:
            self.assertIsNotNone(err, "ValueError")

    def test_double_exponential_smoothing(self):
        """ double_exponential_smoothing юнит-тест """
        date = [4, 6, 4, 6, 4, 6]

        res = double_exponential_smoothing(date, 0.5, 0.5)

        self.assertIsNotNone(res)
        self.assertEqual(res[1:5], [8.0, 7.0, 7.25, 5.5625],
                         "равен (8.0, 7.0, 7.25, 5.5625)")

    def test_get_optimal_params(self):
        """ юнит-тест для get_optimal_params func """
        dataset = get_enriched_dataframe()

        alpha, beta, gamma = get_optimal_params(dataset.sunspots)

        self.assertTrue(alpha > 0., "alpha > 0")
        self.assertTrue(beta >= 0., "beta >= 0")
        self.assertTrue(gamma > 0., "gamma > 0.")

    def test_get_optimal_params2(self):
        """ юнит-тест для get_optimal_params func """
        dataset = get_enriched_dataframe()

        sunspots = pd.Series(data=dataset.sunspots.values[:1200].tolist())
        alpha, beta, gamma = get_optimal_params(sunspots)
        print("alpha, beta, gamma", alpha, beta, gamma)

        self.assertTrue(alpha > 0., "alpha > 0")
        self.assertTrue(beta >= 0., "beta >= 0")
        self.assertTrue(gamma > 0., "gamma > 0.")

    def test_get_optimal_params_negatively1(self):
        """ негативный юнит-тест для get_optimal_params func """
        dataset = np.array([])

        alpha, beta, gamma = get_optimal_params(dataset)

        self.assertEqual(alpha, 0, "alpha == 0")
        self.assertEqual(beta, 0, "beta == 0")
        self.assertEqual(gamma, 0, "gamma == 0")

    def test_hw_exponential_smoothing(self):
        """ юнит-тест для triple exponential smoothing """
        data = get_enriched_dataframe()
        size = len(data["sunspots"].values)

        result = hw_exponential_smoothing(data.sunspots)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), size, f"minimums равен {size}")
        self.assertTrue(result[-1] > 0, " > 0")
