"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    test user
"""
import unittest
from webapp import create_app
from webapp.business_logic import get_user_by_name


class TestUser(unittest.TestCase):
    """ test user """

    def test_get_user_by_name(self):
        """ test get_user_by_name """
        app = create_app()
        with app.app_context():
            user = get_user_by_name("gav")

        self.assertIsNone(user, "user not None")
