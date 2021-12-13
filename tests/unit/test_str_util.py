from adaswarm.utils.strings import str_to_bool
import unittest


class TestStringToBool(unittest.TestCase):
    def test_to_bool_true(self):
        test_string = "True"
        self.assertEqual(True, str_to_bool(test_string))

    def test_to_bool_true_lower_case(self):
        test_string = "true"
        self.assertEqual(True, str_to_bool(test_string))
    
    def test_to_bool_false(self):
        test_string = "False"
        self.assertEqual(False, str_to_bool(test_string))

    def test_to_bool_other(self):
        test_string = "Rabbit"
        with self.assertRaises(ValueError):
            str_to_bool(test_string)