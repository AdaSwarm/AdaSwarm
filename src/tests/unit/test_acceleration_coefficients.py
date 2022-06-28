import os
import unittest
import torch
from adaswarm.particle import AccelerationCoefficients
from unittest.mock import patch


class TestAccelerationCoefficients(unittest.TestCase):
    def tearDown(self):
        os.environ["ADASWARM_DATASET_NAME"] = ""

    def test_set_first_coefficient_default_value_Iris(self):
        os.environ["ADASWARM_DATASET_NAME"] = "Iris"
        coefficients = AccelerationCoefficients()
        self.assertEqual(coefficients.c_1, 0.9)

    def test_set_first_coefficient_default_value_MNIST(self):
        os.environ["ADASWARM_DATASET_NAME"] = "MNIST"
        coefficients = AccelerationCoefficients()

        self.assertEqual(coefficients.c_1, 0.2)
    def test_set_second_coefficient_default_value(self):
        coefficients = AccelerationCoefficients()
        self.assertEqual(coefficients.c_2, 0.8)

    def test_sum_of_coefficients(self):
        os.environ["ADASWARM_DATASET_NAME"] = "MNIST"
        coefficients = AccelerationCoefficients()
        self.assertAlmostEqual(coefficients.sum(), 1.0)

    def test_acceleration_coefficients_scaler(self):
        os.environ["ADASWARM_DATASET_NAME"] = "MNIST"
        coefficients = AccelerationCoefficients()

        with patch("torch.rand", return_value=torch.tensor([0.5])):
            self.assertAlmostEqual(coefficients.random_scale_c_1().item(), 0.1)
            self.assertAlmostEqual(coefficients.random_scale_c_2().item(), 0.4)


if __name__ == "__main__":
    unittest.main()
