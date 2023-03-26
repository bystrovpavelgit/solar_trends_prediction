"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    tests for dl_logic
"""
import unittest
import numpy as np
from webapp.config import RNN_INPUT_SIZE, RNN_OUTPUT_SIZE
from webapp.utils.trends_util import get_lag_fields
from webapp.utils.dataframe_util import sunspot_numbers, prepare_data
from webapp.dl_logic import load_rnn_model, predict_next_cycle, \
    predict_two_cycles, train_lags_dnn_model, calculate_dnn_prediction


class DLModelTest(unittest.TestCase):
    """ tests for dl_logic """
    def test_load_rnn_model_negatively(self):
        """ unit-test for load_rnn_model_negatively """
        filename = "models/RNN.h5"

        model = load_rnn_model(file_name=filename)

        self.assertIsNone(model, "model == None")

    def test_predict_next_cycle(self):
        """ unit-test for predict_next_cycle """
        years, spots = sunspot_numbers().get()
        dummy = np.zeros(RNN_INPUT_SIZE)

        data, times = predict_next_cycle(spots[-RNN_INPUT_SIZE:],
                                         years[-RNN_INPUT_SIZE:])
        data2, times2 = predict_next_cycle(dummy, years[-RNN_INPUT_SIZE:])

        self.assertIsNotNone(data, "predictions not None")
        self.assertIsNotNone(data2, "predictions2 not None")
        self.assertEqual(len(data), RNN_OUTPUT_SIZE)
        self.assertEqual(len(data2), RNN_OUTPUT_SIZE)
        self.assertEqual(len(times), RNN_OUTPUT_SIZE)
        self.assertIsNotNone(times, "times not None")
        self.assertIsNotNone(times2, "times2 not None")
        self.assertTrue(data[0] > 0.1, " > 0")
        self.assertTrue(data[-1] > 0.1, " > 0")
        self.assertNotEqual(data[0], data2[0])
        self.assertNotEqual(data[-1], data2[-1])

    def test_predict_next_cycle_negatively(self):
        """ negative test for predict_next_cycle function """
        dummy = np.zeros(10)
        try:
            predict_next_cycle(dummy, dummy)
        except ValueError as err:
            self.assertIsNotNone(err)

    def test_predict_two_cycles_negatively(self):
        """ negative test for predict_two_cycles function """
        dummy = np.zeros(20)
        try:
            predict_two_cycles(dummy, dummy)
        except ValueError as err:
            self.assertIsNotNone(err)

    def test_train_lags_dnn_model_negatively1(self):
        """ negative test for train_lags_dnn_model function """
        data = None

        self.assertRaises(ValueError, train_lags_dnn_model, data, [])

    def test_train_lags_dnn_model_negatively2(self):
        """ negative test for train_lags_dnn_model function """
        fields1 = None
        fields2 = []

        self.assertRaises(ValueError, train_lags_dnn_model, [0], fields1)
        self.assertRaises(ValueError, train_lags_dnn_model, [0], fields2)

    def test_train_lags_dnn_model_positively(self):
        """ unit-test for train_lags_dnn_model function """
        data = prepare_data()
        lags = get_lag_fields()
        num = len(data["sunspots"].values)

        model = train_lags_dnn_model(data, lags, turns=1)
        trend = model.predict(data[lags].values).reshape(-1)

        self.assertIsNotNone(model)
        self.assertEqual(len(trend), num)
        self.assertTrue(trend[0] > 0.)
        self.assertTrue(trend[-1] > 0.)

    def test_calculate_dnn_prediction_positively(self):
        """ unit-test for calculate_dnn_prediction func """
        data = prepare_data()
        x_train = data[get_lag_fields()].values
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        x_train = (x_train - mean) / std
        y_true = data["sunspots"].values

        predicted, mae = calculate_dnn_prediction(x_train, y_true)

        self.assertIsNotNone(predicted)
        self.assertEqual(len(predicted), len(y_true))
        self.assertTrue(mae > 10.)

    def test_calculate_dnn_prediction_negatively(self):
        """ unit-test for calculate_dnn_prediction func """
        x_data = None
        y_data = None

        self.assertRaises(ValueError, calculate_dnn_prediction, x_data, y_data)
