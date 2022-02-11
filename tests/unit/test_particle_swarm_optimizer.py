import unittest
from numpy.core.fromnumeric import size
from torch import randint, Tensor
from adaswarm.rempso import RotatedEMParticleSwarmOptimizer
from adaswarm.particle import AccelerationCoefficients
from adaswarm.utils.options import get_device
import numpy as np


class TestRotatedEMParticleSwarmOptimizer(unittest.TestCase):
    def test_run_iterations(self):
        number_of_classes = 10
        dimension = 125
        targets = randint(
            low=0,
            high=number_of_classes,
            size=(dimension, number_of_classes),
            device=get_device(),
            requires_grad=False,
        )

        optimizer = RotatedEMParticleSwarmOptimizer(
            targets=np.argmax(targets, axis=1),
            dimension=dimension,
            number_of_classes=number_of_classes,
            swarm_size=6,
            acceleration_coefficients=AccelerationCoefficients(c_1=0.8, c_2=0.9),
        )

        c1r1_c2r2_average, gbest = optimizer.run_iteration(number=5)
        self.assertAlmostEqual(c1r1_c2r2_average, 0.9, places=0)
        self.assertTrue(type(gbest), Tensor)


if __name__ == "__main__":
    unittest.main()
