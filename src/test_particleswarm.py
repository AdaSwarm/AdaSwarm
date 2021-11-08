import unittest
from torch import device as torch_device, randint
from torchswarm.particle import ParticleSwarm, RotatedEMParticle


class TestParticleSwarm(unittest.TestCase):
    def test_default_values(self):
        number_of_classes = 10
        dimension=125
        targets = randint(
                low=0,
                high=number_of_classes,
                size=(dimension, number_of_classes),
                device=torch_device("cpu"),
                requires_grad=False
                )
        swarm = ParticleSwarm(swarm_size=2, targets=targets, dimension=dimension, number_of_classes=number_of_classes)
        self.assertEqual(len(swarm), 2)

    def test_initialise_swarm(self):
        number_of_classes = 10
        dimension=125
        targets = randint(
                low=0,
                high=number_of_classes,
                size=(dimension, number_of_classes),
                device=torch_device("cpu"),
                requires_grad=False
                )
        swarm = ParticleSwarm(swarm_size=2, targets=targets, dimension=dimension, number_of_classes=number_of_classes)
        self.assertIsInstance(swarm[0], RotatedEMParticle)


if __name__ == "__main__":
    unittest.main()
