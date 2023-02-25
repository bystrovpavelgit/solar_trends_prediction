"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    enrich sunspots tests
"""
import unittest
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from webapp.utils.dataframe_util import get_enriched_dataframe
from webapp.utils.enrich_sunspots import predict_using_cross_validation, \
    evaluate_classifier, get_results_for_best_classifier


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
