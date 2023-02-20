""" unit test """
import unittest
from webapp import create_app
from webapp.business_logic import get_user_by_name


class TestUser(unittest.TestCase):
    """ test user """

    def setUp(self):
        """ Инит TestUser """
        self.max_len = 500

    def test_sum(self):
        """ summation юнит-тест """
        self.assertEqual(sum([3, 2]), 5, "равен 5")

    def test_get_user_by_name(self):
        """ test get_user_by_name """
        app = create_app()
        with app.app_context():
            user = get_user_by_name("gav")

        self.assertIsNone(user, "user not None")
