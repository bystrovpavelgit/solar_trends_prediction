"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    Gaps Api unit-tests
"""
import numpy as np
import pandas as pd
import unittest
from sklearn.preprocessing import StandardScaler
from unittest.mock import MagicMock, patch
from webapp.gaps.api import count_gaps, confidence_interval, \
    describe_sunspots_intervals


class GapsApiTest(unittest.TestCase):
    """ Gaps Api unit-tests """

    def test_count_gaps_negatively(self):
        """ negative test for count_gaps """
        data = None

        self.assertRaises(ValueError, count_gaps, data, gap_marker=0.)

    def test_count_gaps(self):
        """ unit-test for count_gaps """
        data = [1.] * 22 + [-1.] * 20

        cnt = count_gaps(pd.Series(data))

        self.assertEqual(cnt, 20)

    def test_count_gaps_mocked(self):
        """ unit-test 2 for count_gaps """
        mock_obj = MagicMock()
        mock_obj.values = np.array([-1.] * 5)

        cnt = count_gaps(mock_obj)

        self.assertEqual(cnt, 5)

    def test_confidence_interval_negatively(self):
        """ negative test for confidence_interval """
        std = -1

        self.assertRaises(ValueError, confidence_interval, std, 0, 0)

    def test_confidence_interval_positively(self):
        """ unit-test for confidence_interval """
        alpha = 0.01
        num = 16
        std = 1.25

        interval = confidence_interval(std, alpha, num)

        self.assertEqual(interval, 0.9208477760433171)

    def test_describe_sunspots_intervals_negatively(self):
        """  negative test for describe_sunspots_intervals """
        data = None

        self.assertRaises(ValueError, describe_sunspots_intervals, data)

    def test_describe_sunspots_intervals_positively(self):
        """ unit-test for describe_sunspots_intervals """
        # mock_function = create_autospec(get_all_minimums, return_value=[8])
        cycle = [0.] + [20.] * 126 + [0.]
        cycle2 = [20.] * 65 + [0.] * 63
        data = np.array((cycle * 5) + (cycle2 * 5), dtype=float)

        res, msg1, msg2, msg3 = describe_sunspots_intervals(data)

        self.assertIsNotNone(res)
        self.assertEqual(res[0], 66)
        self.assertEqual(res[-1], 128)
        self.assertEqual(msg1, "Solar cycle mean duration is 10.092592592592" +
                               "593 years with std. deviation 1.623726682724" +
                               "6648 years")
        self.assertEqual(msg2, "99.4 % Confidence interval for solar cycle " +
                               "duration is [8.087333315832304, 12.097851869" +
                               "352883] (years)")
        self.assertEqual(msg3, "Min sequence length N is 11 for sigma = 1.4," +
                               " alpha = 0.01 and interval = 1.1")

    def test_mock_str(self):
        """ test_mock_str """
        obj = MagicMock()
        obj.__str__.return_value = "Series obj"
        res = str(obj)

        self.assertEqual(res, "Series obj")
        obj.__str__.assert_called()

    def test_standard_scaler(self):
        """ test_standard_scaler """
        trend = [0, 1] * 20

        with patch.object(StandardScaler,
                          'fit_transform',
                          return_value=trend) as mock_method:
            scaler = StandardScaler()
            res = scaler.fit_transform(np.array(list(range(40))))

            mock_method.assert_called()
            self.assertEqual(len(res), 40)
            self.assertEqual(res[0], 0)
            self.assertEqual(res[39], 1)
