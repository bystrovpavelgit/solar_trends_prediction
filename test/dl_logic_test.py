""" tests for dl_logic """
import unittest
import numpy as np
import pandas as pd
from webapp.config import RNN_INPUT_SIZE, RNN_OUTPUT_SIZE
from webapp.dl_logic import load_rnn_model, predict_next_cycle, \
    sunspot_numbers, predict_two_cycles


class DLModelTest(unittest.TestCase):
    """ tests for dl_logic """

    def test_sunspot_numbers(self):
        """ test sunspot_numbers """
        data = pd.read_csv("data/sunspot_numbers.csv", delimiter=";")
        year = data['year_float'].values.tolist()
        spots = data['sunspots'].values.tolist()

        res1, res2 = sunspot_numbers()
        res1, res2 = list(res1), list(res2)

        self.assertIsNotNone(res1, "year not None")
        self.assertIsNotNone(res2, "sunspots not None")
        self.assertEqual(res1, year, "res1 == years")
        self.assertEqual(res2, spots, "res2 == spots")

    def test_load_rnn_model_negatively(self):
        """ unit-test for load_rnn_model_negatively """
        filename = "models/RNN.h5"

        model = load_rnn_model(file_name=filename)

        self.assertIsNone(model, "model == None")

    def test_predict_next_cycle(self):
        """ unit-test for predict_next_cycle """
        years, spots = sunspot_numbers()
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
