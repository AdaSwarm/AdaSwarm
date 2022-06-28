import unittest
from torch import randint, Tensor, tensor, manual_seed
from adaswarm.particle import ParticleSwarm, RotatedEMParticle, AccelerationCoefficients
from adaswarm.utils.options import get_device
from unittest.mock import patch


manual_seed(0)
NUMBER_OF_CLASSES = 10
DIMENSION = 125
coefficients = AccelerationCoefficients()

targets = randint(
    low=0,
    high=NUMBER_OF_CLASSES,
    size=(DIMENSION, NUMBER_OF_CLASSES),
    device=get_device(),
    requires_grad=False,
)

swarm = ParticleSwarm(
    swarm_size=2,
    targets=targets,
    dimension=DIMENSION,
    number_of_classes=NUMBER_OF_CLASSES,
    acceleration_coefficients=coefficients,
    inertial_weight_beta=0.9,
)

class TestParticleSwarm(unittest.TestCase):
    def test_swarm_size(self):
        self.assertEqual(len(swarm), 2)

    def test_initialise_swarm(self):
        self.assertIsInstance(swarm[0], RotatedEMParticle)

    def test_update_velocities(self):
        gbest_position = Tensor(
            size=(DIMENSION, NUMBER_OF_CLASSES), device=get_device()
        )

        gbest_position = gbest_position.fill_(0.3)

        with patch("torch.rand", return_value=tensor([0.5])):
            swarm.update_velocities(gbest_position)

        self.assertAlmostEqual(swarm[0].c_1_r_1, 0.45)
        self.assertAlmostEqual(swarm[1].c_1_r_1, 0.45)
        self.assertAlmostEqual(swarm[0].c_2_r_2, 0.4)
        self.assertAlmostEqual(swarm[1].c_2_r_2, 0.4)

    def test_update_velocities_moving_position(self):
        axis_rotation_factor = 0.4
        patched_list = [x for x in range(2 * int(DIMENSION * axis_rotation_factor))]
        with (
            patch("torch.rand", return_value=tensor([0.5])),
            patch("random.sample", return_value=patched_list),
        ):

            swarm = ParticleSwarm(
                swarm_size=1,
                targets=targets,
                dimension=DIMENSION,
                inertial_weight_beta=0.5,
                number_of_classes=NUMBER_OF_CLASSES,
                acceleration_coefficients=coefficients
            )
            gbest_position = tensor([[-4] * NUMBER_OF_CLASSES] * DIMENSION)
            swarm.update_velocities(gbest_position)
            self.assertAlmostEqual(swarm[0].position[0][0].item(), -0.0416, places=3)

    def test_calculate_swarm_scaled_coeffiecient_average(self):
        gbest_position = Tensor(
            size=(DIMENSION, NUMBER_OF_CLASSES), device=get_device()
        )

        gbest_position = gbest_position.fill_(0.3)

        with patch("torch.rand", return_value=tensor([0.5])):
            swarm.update_velocities(gbest_position=gbest_position)

        self.assertAlmostEqual(
            swarm.average_of_scaled_acceleration_coefficients(), 0.85
        )

        # self.assertAlmostEqual(swarm.sum_of_c_2_r_2(), 0.8)


if __name__ == "__main__":
    unittest.main()
