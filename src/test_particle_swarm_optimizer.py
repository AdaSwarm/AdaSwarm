import unittest
from torchswarm.rempso import RotatedEMParticleSwarmOptimizer


class TestRotatedEMParticleSwarmOptimizer(unittest.TestCase):
    def test_run_iterations(self):
        optimizer = RotatedEMParticleSwarmOptimizer()
        c1r1, c2r2, gbest = optimizer.run_iteration(number=5)
        self.assertEqual(c1r1, 2)
        self.assertEqual(c2r2, 2)
        self.assertEqual(gbest, 2)


if __name__ == "__main__":
    unittest.main()
