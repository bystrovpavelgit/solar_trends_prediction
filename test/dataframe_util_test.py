"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    tests for dataframe utility
"""
import unittest
import numpy as np
import pandas as pd
from webapp.utils.dataframe_util import fill_values, rolling_mean, \
    get_enriched_dataframe, sunspot_numbers, \
    get_all_minimums, find_minimum_with_strategy


class DataframeUtilTest(unittest.TestCase):
    """ tests for dataframe utility """
    def test_fill_values(self):
        """ test fill values """
        data = np.array([1, 3, 1, 3, 1, 3, 1, 3, 1, 3])
        indices = [0, 6, 10]

        filled = fill_values(data, indices, np.mean)

        self.assertEqual(len(filled), 10, "len equals 10")
        self.assertEqual(filled[0], 2., "mean equals 2")
        self.assertEqual(filled[9], 2., "mean equals 2")

    def test_fill_values_negatively(self):
        """ test fill values """
        data = np.array([1])
        indices = [0]

        filled = fill_values(data, indices, np.mean)

        self.assertEqual(filled, np.array([1]), "len equals 1")

    def test_fill_values_negatively2(self):
        """ test fill values """
        data = np.array([])
        indices = []

        filled = fill_values(data, indices, np.mean)

        self.assertEqual(len(filled), 0, "empty")

    def test_fill_values_negatively3(self):
        """ test fill values """
        data = np.array([1, 3, 1, 3, 1, 3, 1, 3, 1, 3])
        indices = [0, 22]

        filled = fill_values(data, indices, np.mean)

        self.assertEqual(len(filled), 0, "empty")

    def test_get_enriched_dataframe(self):
        """
            test get_enriched_dataframe function using
            csv_file="data/sunspot_numbers.csv"
        """
        csv = "data/sunspot_numbers.csv"

        data = get_enriched_dataframe(csv_file=csv)

        self.assertTrue("year_float" in data.columns)
        self.assertTrue("Year" in data.columns)
        self.assertTrue("sunspots" in data.columns)
        self.assertTrue("observations" in data.columns)
        self.assertTrue("mean_1y" in data.columns)
        self.assertTrue("mean_3y" in data.columns)
        self.assertTrue("mean_12y" in data.columns)
        self.assertTrue("sn_mean" in data.columns)
        self.assertTrue("sn_max" in data.columns)
        self.assertTrue("sn_min" in data.columns)
        self.assertTrue("y_min" in data.columns)
        self.assertTrue("y_max" in data.columns)

    def test_get_enriched_dataframe_negatively(self):
        """ test get_enriched_dataframe function
            csv_file="data/sunspot_numbers.csv"
        """
        csv = "none.csv"

        self.assertRaises(FileNotFoundError, get_enriched_dataframe, csv_file=csv)

    def test_sunspot_numbers(self):
        """ test sunspot_numbers """
        data = pd.read_csv("data/sunspot_numbers.csv", delimiter=";")
        year = data['year_float'].values.tolist()
        spots = data['sunspots'].values.tolist()

        res1, res2 = sunspot_numbers().get()
        res1, res2 = list(res1), list(res2)

        self.assertIsNotNone(res1, "year not None")
        self.assertIsNotNone(res2, "sunspots not None")
        self.assertEqual(res1, year, "res1 == years")
        self.assertEqual(res2, spots, "res2 == spots")

    def test_rolling_mean(self):
        """ rolling_mean юнит-тест """
        date = pd.Series([0, 4, 6, 4, 6, 4, 6])

        stat = rolling_mean(date, 6).values.tolist()

        self.assertIsNotNone(stat)
        self.assertEqual(stat[5], 4., "равен 4")
        self.assertEqual(stat[6], 5., "равен 5")

    def test_min_index_positively1(self):
        """ позитивный юнит-тест для min_index """
        arr = np.array([2, 3, 5, 5, 3, 1, 2])

        index = find_minimum_with_strategy(arr, 0, 6, strategy=1)

        self.assertEqual(index, 5, "min_index = 5")

    def test_min_index_positively2(self):
        """ позитивный юнит-тест для min_index """
        arr = np.array([2, 3, 6, 5, 3, 1, 9])

        index = find_minimum_with_strategy(arr, 0, 6, strategy=0)

        self.assertEqual(index, 0, "min_index = 0")

    def test_min_index_negatively(self):
        """ негативный юнит-тест для min_index """
        empty = np.array([])

        self.assertRaises(ValueError, find_minimum_with_strategy, empty, 0, 9)

    def test_get_all_minimums_positively(self):
        """ негативный юнит-тест для find_minimums """
        inp = np.array([2, 3, 5, 5, 3, 1, 2])

        arr = get_all_minimums(inp, 7)

        self.assertEqual(arr, [5], "minimums равен [5]")

    def test_get_all_minimums_negatively1(self):
        """ негативный юнит-тест для find_minimums """
        empty = np.array([])

        arr = get_all_minimums(empty, 9)

        self.assertEqual(arr, [], "minimums = []")

    def test_get_all_minimums_negatively2(self):
        """ негативный юнит-тест для find_minimums """
        empty = None

        arr = get_all_minimums(empty, 5)

        self.assertEqual(arr, [], "minimums = []")

    def test_get_all_minimums_negatively3(self):
        """ негативный юнит-тест для find_minimums """
        arr = np.array([2, 3, 5])

        arr = get_all_minimums(arr, -1)

        self.assertEqual(arr, [], "minimums = []")

    def test_get_all_minimums_negatively4(self):
        """ негативный юнит-тест для find_minimums """
        arr = np.array([2, 3, 5])

        arr = get_all_minimums(arr, 0)

        self.assertEqual(arr, [], "minimums = []")
