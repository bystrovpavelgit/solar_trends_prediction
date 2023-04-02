"""
    Apache License 2.0 Copyright (c) 2023 Pavel Bystrov
    plot utility tests
"""
import pandas as pd
import numpy as np
import unittest
from webapp.utils.plot_util import random_uuid, autocorr_image


class PlotUtilTest(unittest.TestCase):
    """ plot utility tests """
    def test_random_uuid(self):
        """ unit-test for random_uuid """
        id1 = random_uuid()
        id2 = random_uuid()

        self.assertTrue(id1 > -1)
        self.assertTrue(id2 > -1)
        self.assertNotEqual(id1, id2)

    def test_autocorr_image(self):
        """ unit-test for autocorr_image """
        data = pd.Series(np.ones(512).tolist() + [0., 0.])

        file_name, p_value = autocorr_image(data)

        self.assertTrue(file_name.startswith("webapp/static"))
        self.assertTrue(file_name.endswith(".jpg"))
        self.assertTrue(p_value >= 0.)
