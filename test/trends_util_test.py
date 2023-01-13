import unittest
from webapp.utils.trends_util import min_index, find_minimums, moving_average


class TrendsUtilTest(unittest.TestCase):
    """ Trends Util test """
    def setUp(self):
        """ Инит TestUser """
        self.max_len = 500

    def test_sum(self):
        """ summation юнит-тест """
        print("юнит-тест sum\n")
        self.assertEqual(sum([3, 2]), 5, "равен 5")

    def test_min_index(self):
        """ min_index юнит-тест """
        self.assertEqual(sum([3, 2]), 5, "равен 5")
