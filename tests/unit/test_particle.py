import unittest
from torch import randint, device as torch_device, manual_seed, tensor
from adaswarm.particle import RotatedEMParticle, AccelerationCoefficients
from unittest.mock import patch


class TestRotatedEMParticleAttributes(unittest.TestCase):
    def test_particle_scaled_acceleration_coefficients(self):
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

        with patch("torch.rand", return_value=tensor([0.5])):
            self.assertAlmostEqual(
                coefficients.random_scale_c_1().item(), 0.35)
            self.assertAlmostEqual(coefficients.random_scale_c_2().item(), 0.2)


            particle_this = RotatedEMParticle(
                number_of_classes=number_of_classes, dimensions=dimension,
                beta=beta, acceleration_coefficients=coefficients,
                targets=targets, device=device)
            gbest_position = tensor([[-4] * number_of_classes] * dimension)
            particle_this.update_velocity(gbest_position)
            self.assertAlmostEqual(particle_this.c_1_r_1, 0.35)
            self.assertAlmostEqual(particle_this.c_2_r_2, 0.2)


if __name__ == "__main__":
    unittest.main()
