"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    tests for deep learning logic
"""
import unittest
from webapp.stat.api import get_smoothed_data_by_type
from webapp.utils.dataframe_util import get_enriched_dataframe


class ViewsTest(unittest.TestCase):
    """ tests for dl_logic """

    def test_get_smoothed_data_by_type_negatively(self):
        """ test get_smoothed_data_by_type """
        result = get_smoothed_data_by_type("")

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], [])
        self.assertEqual(result[1], [])
        self.assertEqual(result[2], [])

    def test_get_smoothed_data_by_type_positively(self):
        """ test get_smoothed_data_by_type """
        data = get_enriched_dataframe()
        time = data["year_float"].values.tolist()
        sunspots = data["sunspots"].values.tolist()
        smoothed12 = data["mean_12y"].values.tolist()
        smoothed3 = data["mean_3y"].values.tolist()
        type1 = "скользязщее среднее 3г"
        type2 = "скользязщее среднее 12л"

        result1 = get_smoothed_data_by_type(type1)
        result2 = get_smoothed_data_by_type(type2)

        self.assertEqual(len(result1), 3)
        self.assertEqual(len(result2), 3)
        self.assertEqual(len(result1[0]), len(time))
        self.assertEqual(len(result2[0]), len(time))
        self.assertEqual(len(result1[1]), len(sunspots))
        self.assertEqual(len(result2[1]), len(sunspots))
        self.assertEqual(len(result1[2]), len(smoothed3))
        self.assertEqual(len(result2[2]), len(smoothed12))
        self.assertEqual(result1[2][0], smoothed3[0])
        self.assertEqual(result1[2][-1], smoothed3[-1])
        self.assertEqual(result2[2][0], smoothed12[0])
        self.assertEqual(result2[2][-1], smoothed12[-1])

    def test_get_smoothed_data_by_type_negatively2(self):
        """ negative unit-test get_smoothed_data_by_type """
        type_ = None

        self.assertRaises(ValueError, get_smoothed_data_by_type, type_)
