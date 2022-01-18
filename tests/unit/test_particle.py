import unittest
from torch import randint, device as torch_device, manual_seed, tensor
from adaswarm.particle import RotatedEMParticle, AccelerationCoefficients
from unittest.mock import patch

manual_seed(0)
number_of_classes = 10
dimension = 125
beta = 0.5
coefficients = AccelerationCoefficients(c_1=0.7, c_2=0.4)
device = torch_device("cpu")
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
            self.assertAlmostEqual(coefficients.random_scale_c_1().item(), 0.35)
            self.assertAlmostEqual(coefficients.random_scale_c_2().item(), 0.2)


if __name__ == "__main__":
    unittest.main()
