import unittest
from torch import randint, device as torch_device, manual_seed, tensor
from adaswarm.particle import RotatedEMParticle, AccelerationCoefficients, update_velocity
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

            particle_this = RotatedEMParticle(
                number_of_classes=number_of_classes,
                dimensions=dimension,
                beta=beta,
                acceleration_coefficients=coefficients,
                targets=targets,
                device=device,
            )
            gbest_position = tensor([[-4] * number_of_classes] * dimension)
            update_velocity(gbest_position, particle_this)
            self.assertAlmostEqual(particle_this.c_1_r_1, 0.35)
            self.assertAlmostEqual(particle_this.c_2_r_2, 0.2)

    def test_update_velocity_moving_position(self):
        axis_rotation_factor = 0.4
        patched_list = [x for x in range(2 * int(dimension * axis_rotation_factor))]
        with (
            patch("torch.rand", return_value=tensor([0.5])),
            patch("random.sample", return_value=patched_list),
        ):

            particle_this = RotatedEMParticle(
                number_of_classes=number_of_classes,
                dimensions=dimension,
                beta=beta,
                acceleration_coefficients=coefficients,
                targets=targets,
                device=device,
            )
            gbest_position = tensor([[-4] * number_of_classes] * dimension)
            update_velocity(gbest_position, particle_this)
            self.assertAlmostEqual(particle_this.position[0][0].item(), 0.729, places=3)


if __name__ == "__main__":
    unittest.main()
