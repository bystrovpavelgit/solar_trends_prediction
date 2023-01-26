""" enrich sunspots tests """
import unittest
import numpy as np
from webapp.utils.enrich_sunspots import fill_values, \
    get_enriched_dataframe


class EnrichSunspotsTest(unittest.TestCase):
    """ enrich sunspots tests """
    def setUp(self):
        """ set Up"""
        self.num = 5

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
        """ test get_enriched_dataframe function csv_file="data/sunspot_numbers.csv" """
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
