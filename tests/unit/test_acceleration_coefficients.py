import unittest
from adaswarm.particle import AccelerationCoefficients


class TestAccelerationCoefficients(unittest.TestCase):
    def test_set_first_coefficient_default_value(self):
        coefficients = AccelerationCoefficients()
        self.assertEqual(coefficients.c_1, 0.9)

    def test_set_second_coefficient_default_value(self):
        coefficients = AccelerationCoefficients()
        self.assertEqual(coefficients.c_2, 0.8)

    def test_set_first_coefficient_value(self):
        coefficients = AccelerationCoefficients(c_1=0.3)
        self.assertEqual(coefficients.c_1, 0.3)

    def test_set_second_coefficient_value(self):
        coefficients = AccelerationCoefficients(c_2=0.4)
        self.assertEqual(coefficients.c_2, 0.4)

    def test_sum_of_coefficients(self):
        coefficients = AccelerationCoefficients(c_1=0.7, c_2=0.4)
        self.assertEqual(coefficients.sum(), 1.1)


if __name__ == "__main__":
    unittest.main()
