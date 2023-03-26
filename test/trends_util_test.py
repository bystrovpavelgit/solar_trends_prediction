"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    Trends Util unit-tests
"""
import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from webapp.utils.dataframe_util import get_enriched_dataframe, prepare_data
from webapp.utils.trends_util import prediction_by_type, get_optimal_params, \
    double_exponential_smoothing, exponential_smoothing, \
    hw_exponential_smoothing, regression_prediction


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
        """ exponential_smoothing юнит-тест 2 """
        input1 = [4]
        input2 = -0.5

        self.assertRaises(ValueError, exponential_smoothing, input1, input2)

    def test_exponential_smoothing_negatively3(self):
        """ exponential_smoothing юнит-тест 3 """
        input1 = []
        input2 = 0.1

        self.assertRaises(ValueError, exponential_smoothing, input1, input2)

    def test_double_exponential_smoothing(self):
        """ double_exponential_smoothing юнит-тест """
        date = [4, 6, 4, 6, 4, 6]

        res = double_exponential_smoothing(date, 0.5, 0.5)

        self.assertIsNotNone(res)
        self.assertEqual(res[1:5], [8.0, 7.0, 7.25, 5.5625],
                         "равен (8.0, 7.0, 7.25, 5.5625)")

    def test_double_exponential_smoothing_negatively2(self):
        """ double_exponential_smoothing юнит-тест """

        self.assertRaises(ValueError,
                          double_exponential_smoothing,
                          [4],
                          -0.1,
                          -0.5)

    def test_double_exponential_smoothing_negatively3(self):
        """ double_exponential_smoothing юнит-тест """
        self.assertRaises(ValueError, double_exponential_smoothing, [], .1, .5)

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
        """ юнит-тест для hw_exponential_smoothing """
        data = get_enriched_dataframe()
        size = len(data["sunspots"].values)

        result = hw_exponential_smoothing(data.sunspots)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), size, f"minimums равен {size}")
        self.assertTrue(result[-1] > 0, " > 0")

    def test_hw_exponential_smoothing_negatively2(self):
        """ юнит-тест 2 для hw_exponential_smoothing """
        data = pd.Series([])

        self.assertRaises(ValueError, hw_exponential_smoothing,
                          data, sess_len=1)

    def test_hw_exponential_smoothing_negatively3(self):
        """ юнит-тест 3 для hw_exponential_smoothing """
        data = pd.Series([1])

        self.assertRaises(ValueError, hw_exponential_smoothing,
                          data, sess_len=-1)

    def test_prediction_by_type_linear(self):
        """ юнит-тест для prediction by type linear """
        data = prepare_data()

        predicted, mae = prediction_by_type("Linear", data)

        self.assertEqual(predicted[0], 8.225963980612107)
        self.assertEqual(mae, 17.399226421749635)

    def test_prediction_by_type_ridge(self):
        """ юнит-тест для prediction by type ridge """
        data = prepare_data()

        predicted, mae = prediction_by_type("Ridge", data)

        self.assertEqual(predicted[0], 8.228347021413697)
        self.assertEqual(mae, 17.3993339765839)

    def test_prediction_by_type_negative(self):
        """ negative юнит-тест для prediction by type """
        data = None

        self.assertRaises(ValueError, prediction_by_type, "", data)

    def test_regression_prediction_linear(self):
        """ юнит-тест для regression_prediction """
        data = prepare_data()
        reg = LinearRegression()

        predicted, mae = regression_prediction(reg, data)

        self.assertEqual(predicted[0], 8.225963980612107)
        self.assertEqual(mae, 17.399226421749635)

    def test_regression_prediction_ridge(self):
        """ юнит-тест для regression_prediction """
        data = prepare_data()
        reg = Ridge(alpha=0.2)

        prediction, mae = regression_prediction(reg, data)

        self.assertEqual(prediction[0], 8.228347021413697)
        self.assertEqual(mae, 17.3993339765839)

    def test_regression_prediction_negative(self):
        """ negative юнит-тест для regression_prediction """
        data = None

        self.assertRaises(ValueError, regression_prediction, None, data)
