import unittest
from torchswarm.rempso import ParticleSwarm

class TestParticleSwarm(unittest.TestCase):

    def test_default_values(self):
        
        my_particle_swarm = ParticleSwarm()
        self.assertEqual(len(my_particle_swarm.swarm), 100)


if __name__ == '__main__':
    unittest.main()