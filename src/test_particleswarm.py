import unittest
from torchswarm.particle import ParticleSwarm, RotatedEMParticle


class TestParticleSwarm(unittest.TestCase):
    def test_default_values(self):
        swarm = ParticleSwarm(swarm_size=2)
        self.assertEqual(len(swarm), 2)

    def test_initialise_swarm(self):
        swarm = ParticleSwarm(swarm_size=2)
        self.assertIsInstance(swarm[0], RotatedEMParticle)


if __name__ == "__main__":
    unittest.main()
