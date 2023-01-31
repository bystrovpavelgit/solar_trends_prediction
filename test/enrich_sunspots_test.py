""" enrich sunspots tests """
import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from webapp.utils.enrich_sunspots import fill_values, \
    get_enriched_dataframe, predict_cv_and_plot_results


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

    def test_get_enriched_dataframe_negatively(self):
        """ test get_enriched_dataframe function csv_file="data/sunspot_numbers.csv" """
        csv = "none.csv"

        try:
            get_enriched_dataframe(csv_file=csv)
        except FileNotFoundError as err:
            self.assertEqual(err.strerror, "No such file or directory")

    def test_predict_cv_and_plot_results(self):
        """ test_predict_cv_and_plot_results """
        clsf = RandomForestClassifier()
        params = {"n_estimators": [3, 4, 5, 7], "max_depth": [3, 4, 5, 6, 9]}
        dframe = get_enriched_dataframe()
        cols = ["sunspots", "observations", "mean_1y", "mean_3y", "mean_12y",
                "sn_mean", "sn_max", "sn_min"]
        data_scaled = StandardScaler().fit_transform(dframe[cols].values)

        score, best_params = predict_cv_and_plot_results(clsf,
                                                         params,
                                                         data_scaled,
                                                         dframe)

        self.assertTrue(score > 0.9)
        self.assertTrue(best_params['max_depth'] >= 4)
        self.assertTrue(best_params['n_estimators'] >= 3)
