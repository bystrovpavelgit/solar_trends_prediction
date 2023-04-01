"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    Gaps Api unit-tests
"""
import unittest
from webapp.gaps.api import count_gaps, confidence_interval, \
    describe_sunspots_intervals


class GapsApiTest(unittest.TestCase):
    """ Gaps Api unit-tests """
    def test_count_gaps_negatively(self):
        """ negative test for count_gaps """
        data = None

        self.assertRaises(ValueError, count_gaps, data, gap_marker=0.)

    def test_confidence_interval_negatively(self):
        """ negative test for confidence_interval """
        std = -1

        self.assertRaises(ValueError, confidence_interval, std, 0, 0)

    def test_describe_sunspots_intervals_negatively(self):
        """  negative test for describe_sunspots_intervals """
        data = None

        self.assertRaises(ValueError, describe_sunspots_intervals, data)
