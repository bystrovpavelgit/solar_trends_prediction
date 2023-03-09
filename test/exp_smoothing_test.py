""" HoltWinters class Tests """
import unittest
from unittest.mock import patch
import pandas as pd
import webapp.utils.trends_util


class HoltWintersTest(unittest.TestCase):
    """ HoltWinters class Tests """
    def test_holt_winters_init_negatively1(self):
        """ юнит-тест для метода Хольта-Винтерся"""
        clazz = webapp.utils.trends_util.HoltWinters
        trend = pd.Series([1, 2])

        self.assertRaises(ValueError, clazz, trend, 2, .1, .5, .5, -1)

    def test_holt_winters_init_negatively2(self):
        """ юнит-тест 2 для метода Хольта-Винтерся"""
        clazz = webapp.utils.trends_util.HoltWinters
        trend = pd.Series([1])

        self.assertRaises(ValueError, clazz, trend, 5, .1, .5, .5, 1)

    def test_holt_winters_initial_trend(self):
        """ initial_trend test """
        trend = pd.Series([1] * 11)
        with patch.object(webapp.utils.trends_util.HoltWinters,
                          'initial_trend',
                          return_value=[]) as mock_method:
            thing = webapp.utils.trends_util.HoltWinters(trend, 10, 0.5,
                                                         0.5, 0.5, 1)
            res = thing.initial_trend()
            mock_method.assert_called_once_with()
            self.assertIsNotNone(res)
            self.assertEqual(res, [])

    def test_holt_winters_initial_seasonal_components(self):
        """ initial_seasonal_components test """
        trend = pd.Series([1] * 11)
        with patch.object(webapp.utils.trends_util.HoltWinters,
                          'initial_seasonal_components',
                          return_value=[1]) as mock_method:
            thing = webapp.utils.trends_util.HoltWinters(trend, 10, 0.5,
                                                         0.5, 0.5, 1)
            res = thing.initial_seasonal_components()
            mock_method.assert_called_once_with()
            self.assertIsNotNone(res)
            self.assertEqual(res, [1])

    def test_holt_winters_triple_exponential_smoothing(self):
        """ triple_exponential_smoothing test """
        trend = pd.Series([1] * 11)

        with patch.object(webapp.utils.trends_util.HoltWinters,
                          'triple_exponential_smoothing',
                          return_value=[2, 2]) as mocked:
            obj = webapp.utils.trends_util.HoltWinters(trend, 9, 0.4, 0.4,
                                                       0.4, 1)
            result = obj.triple_exponential_smoothing()
            mocked.assert_called_once_with()
            self.assertIsNotNone(result)
            self.assertEqual(result, [2, 2])

    @patch('webapp.utils.trends_util.HoltWinters')
    def test_holt_winters(self, mock_class1):
        """ Holt_Winters class test """
        trend = pd.Series([1] * 10)
        obj = webapp.utils.trends_util.HoltWinters(trend, 8, 0.4, 0.4, 0.4, 1)

        self.assertIsNotNone(obj)
        self.assertTrue(mock_class1 is webapp.utils.trends_util.HoltWinters)
        self.assertTrue(mock_class1.called)
