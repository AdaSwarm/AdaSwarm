from typing import Tuple
import unittest
from torch import device as torch_device, randint, Tensor, manual_seed
from adaswarm.particle import ParticleSwarm, RotatedEMParticle


class TestParticleSwarm(unittest.TestCase):
    def test_swarm_size(self):
        number_of_classes = 10
        dimension = 125
        targets = randint(
            low=0,
            high=number_of_classes,
            size=(dimension, number_of_classes),
            device=torch_device("cpu"),
            requires_grad=False
        )
        swarm = ParticleSwarm(swarm_size=2, targets=targets,
                              dimension=dimension, number_of_classes=number_of_classes)
        self.assertEqual(len(swarm), 2)

    def test_initialise_swarm(self):
        number_of_classes = 10
        dimension = 125
        targets = randint(
            low=0,
            high=number_of_classes,
            size=(dimension, number_of_classes),
            device=torch_device("cpu"),
            requires_grad=False
        )
        swarm = ParticleSwarm(swarm_size=2, targets=targets,
                              dimension=dimension, number_of_classes=number_of_classes)
        self.assertIsInstance(swarm[0], RotatedEMParticle)

    def test_update_velocities(self):
        manual_seed(0)
        number_of_classes = 10
        dimension = 125
        targets = randint(
            low=0,
            high=number_of_classes,
            size=(dimension, number_of_classes),
            device=torch_device("cpu"),
            requires_grad=False
        )

        swarm = ParticleSwarm(swarm_size=2, targets=targets,
                              dimension=dimension, number_of_classes=number_of_classes)
        gbest_position = Tensor(
            size=(dimension, number_of_classes), device=torch_device("cpu")
        )

        gbest_position = gbest_position.fill_(0.3)
        c1r1_list, c2r2_list = swarm.update_velocities(gbest_position)
        self.assertEqual(c1r1_list[0], 0.7587156891822815)
        self.assertEqual(c1r1_list[1], 0.3397567868232727)
        self.assertEqual(c2r2_list[0], 0.5604957938194275)
        self.assertEqual(c2r2_list[1], 0.6286374926567078)


if __name__ == "__main__":
    unittest.main()
