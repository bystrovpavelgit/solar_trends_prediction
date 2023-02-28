""" MockTest """
import unittest
import pandas as pd
from unittest.mock import patch
import webapp.utils.trends_util


class HoltWintersTest(unittest.TestCase):
    """ HoltWinters method Test """
    def test_holt_winters_initial_trend(self):
        """ initial_trend test """
        trend = pd.Series([1] * 11)
        with patch.object(webapp.utils.trends_util.HoltWinters,
                          'initial_trend',
                          return_value=[]) as mock_method:
            thing = webapp.utils.trends_util.HoltWinters(trend, 10, 0.5, 0.5, 0.5, 1)
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
            thing = webapp.utils.trends_util.HoltWinters(trend, 10, 0.5, 0.5, 0.5, 1)
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
            obj = webapp.utils.trends_util.HoltWinters(trend, 9, 0.4, 0.4, 0.4, 1)
            result = obj.triple_exponential_smoothing()
            mocked.assert_called_once_with()
            self.assertIsNotNone(result)
            self.assertEqual(result, [2, 2])

# @patch('module.ClassName2')
# @patch('module.ClassName1')
# def test(MockClass1, MockClass2):
#     module.ClassName1()
#     module.ClassName2()
#     assert MockClass1 is module.ClassName1
#     assert MockClass2 is module.ClassName2
#     assert MockClass1.called
#     assert MockClass2.called
