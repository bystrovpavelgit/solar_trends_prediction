"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    gaps utility tests
"""
import unittest
import numpy as np
from fedot.core.pipelines.pipeline import Pipeline
from webapp.utils.gaps_util import fill_gaps, get_simple_pipeline, \
    get_composite_pipeline


class GapsFillTest(unittest.TestCase):
    """ юнит-тесты для gaps_util.py """
    def test_fill_gaps_negatively(self):
        """ негативный юнит-тест для fill_gaps """
        new_path = "data/empty.csv"

        self.assertRaises(FileNotFoundError, fill_gaps, file_path=new_path)

    def test_fill_gaps_positively(self):
        """ юнит-тест для fill_gaps """
        data = fill_gaps()

        self.assertIsNotNone(data["ridge"])
        self.assertIsNotNone(data["composite"])
        self.assertTrue(len(data["ridge"].values) > 0)
        self.assertTrue(len(data["composite"].values) > 0)
        self.assertTrue(np.min(data["composite"].values) > -1.)
        self.assertTrue(np.min(data["ridge"].values) > -1.)
        self.assertTrue(np.min(data["with_gap"].values),  -1.)

    def test_get_simple_pipeline(self):
        """ юнит-тест для get_simple_pipeline """
        pipe = get_simple_pipeline()

        self.assertIsNotNone(pipe)
        self.assertTrue(isinstance(pipe, Pipeline))

    def test_get_composite_pipeline(self):
        """ юнит-тест для get_composite_pipeline """
        pipe = get_composite_pipeline()

        self.assertIsNotNone(pipe)
        self.assertTrue(isinstance(pipe, Pipeline))
