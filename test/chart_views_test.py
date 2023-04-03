"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    юнит-тесты для chart/views.py
"""
import unittest
from webapp.chart.views import count_sunspots_data
from webapp.utils.dataframe_util import get_enriched_dataframe


class ChartViewTest(unittest.TestCase):
    """ юнит-тесты для chart/views.py """
    def test_count_sunspots_data(self):
        """ count_sunspots_data юнит-тест """
        data = get_enriched_dataframe()

        result = count_sunspots_data(data)

        self.assertEqual(len(result), 4, "size == 4")
        self.assertEqual(result, [612, 1200, 1200, 275])
