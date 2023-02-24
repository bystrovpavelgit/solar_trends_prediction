""" enrich sunspots tests """
import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from webapp.utils.enrich_sunspots import fill_values, \
    get_enriched_dataframe, predict_using_cross_validation, \
    evaluate_classifier, get_results_for_best_classifier, \
    get_users_timeseries


class EnrichSunspotsTest(unittest.TestCase):
    """ enrich sunspots tests """

    def setUp(self):
        """ set Up"""
        self.params = {"n_estimators": [3, 4, 5, 7],
                       "max_depth": [3, 4, 5, 6, 9]}
        self.dframe = get_enriched_dataframe()
        cols = ["sunspots", "observations", "mean_1y", "mean_3y", "mean_12y",
                "sn_mean", "sn_max", "sn_min"]
        self.data_scaled = StandardScaler().fit_transform(
            self.dframe[cols].values)

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

        try:
            get_enriched_dataframe(csv_file=csv)
        except FileNotFoundError as err:
            self.assertEqual(err.strerror, "No such file or directory")

    def test_predict_using_cross_validation1(self):
        """ test_predict_cv_and_plot_results """
        clsf = RandomForestClassifier()
        scaled = self.data_scaled

        score, best_params = predict_using_cross_validation(clsf,
                                                            self.params,
                                                            scaled,
                                                            self.dframe)

        self.assertTrue(score > 0.92)
        self.assertTrue(best_params['max_depth'] >= 4)
        self.assertTrue(best_params['n_estimators'] >= 3)

    def test_predict_using_cross_validation2(self):
        """ test_predict_cv_and_plot_results """
        clsf = XGBClassifier(random_state=17)
        scaled = self.data_scaled

        score, best_params = predict_using_cross_validation(clsf,
                                                            self.params,
                                                            scaled,
                                                            self.dframe)

        self.assertTrue(score > 0.91)
        self.assertTrue(best_params['max_depth'] >= 3)
        self.assertTrue(best_params['n_estimators'] >= 3)

    def test_evaluate_classifier(self):
        """ test evaluate_classifier function"""
        params = {"n_estimators": 4, "max_depth": 3}
        clsf = ExtraTreesClassifier()

        predict_max, predict_min, max_, sunspots = \
            evaluate_classifier(clsf,
                                params,
                                self.data_scaled,
                                self.dframe)

        self.assertIsNotNone(predict_max)
        self.assertIsNotNone(predict_min)
        self.assertIsNotNone(max_)
        self.assertIsNotNone(sunspots)

    def test_evaluate_classifier_negatively1(self):
        """ negatively test evaluate_classifier function"""
        params = {"n_estimators": 4, "max_depth": 3}
        clsf = None
        try:
            res = \
                evaluate_classifier(clsf,
                                    params,
                                    self.data_scaled,
                                    self.dframe)
            self.assertIsNotNone(res)
        except ValueError as err:
            self.assertIsNotNone(err)

    def test_evaluate_classifier_negatively2(self):
        """ negatively test evaluate_classifier function"""
        params = None
        clsf = ExtraTreesClassifier()
        try:
            res = \
                evaluate_classifier(clsf,
                                    params,
                                    self.data_scaled,
                                    self.dframe)
            self.assertIsNotNone(res)
        except ValueError as err:
            self.assertIsNotNone(err)

    def test_get_results_for_best_classifier(self):
        """ test get_results_for_best_classifier """
        results = get_results_for_best_classifier()

        self.assertIsNotNone(results)
        self.assertEqual(len(results), 5)
        self.assertEqual(len(results[0]), len(results[1]))
        self.assertEqual(len(results[0]), len(results[2]))
        self.assertEqual(len(results[0]), len(results[3]))
        self.assertEqual(len(results[0]), len(results[4]))

    def test_get_users_timeseries_negatively(self):
        """ test get_users_timeseries """
        csv = "data/none.csv"

        data = get_users_timeseries(csv_file=csv)

        self.assertIsNotNone(data)
        self.assertTrue(data.is_empty())
