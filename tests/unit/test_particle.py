import unittest
from torch import randint, manual_seed, tensor
from adaswarm.particle import AccelerationCoefficients
from adaswarm.utils.options import get_device
from unittest.mock import patch

manual_seed(0)
number_of_classes = 10
dimension = 125
beta = 0.5
coefficients = AccelerationCoefficients()
device = get_device()
targets = randint(
    low=0,
    high=number_of_classes,
    size=(dimension, number_of_classes),
    device=device,
    requires_grad=False,
)


class TestRotatedEMParticleAttributes(unittest.TestCase):
    def test_particle_scaled_acceleration_coefficients(self):

        with patch("torch.rand", return_value=tensor([0.5])):
            self.assertAlmostEqual(coefficients.random_scale_c_1().item(), 0.45)
            self.assertAlmostEqual(coefficients.random_scale_c_2().item(), 0.4)


if __name__ == "__main__":
    unittest.main()
