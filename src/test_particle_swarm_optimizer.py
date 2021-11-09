import unittest
from torch import randint, device as torch_device, Tensor
from torch.autograd import Variable
from torchswarm.rempso import RotatedEMParticleSwarmOptimizer
import numpy as np

class TestRotatedEMParticleSwarmOptimizer(unittest.TestCase):
    def test_run_iterations(self):
        number_of_classes = 10
        dimension=125
        targets = randint(
                low=0,
                high=number_of_classes,
                size=(dimension, number_of_classes),
                device=torch_device("cpu"),
                requires_grad=False
                )

        optimizer = RotatedEMParticleSwarmOptimizer(
            targets=np.argmax(targets,axis=1), 
            dimension=dimension, 
            number_of_classes=number_of_classes
        )
        
        c1r1, c2r2, gbest = optimizer.run_iteration(number=5)
        self.assertAlmostEqual(c1r1, 1.0, places=0)
        self.assertAlmostEqual(c2r2, 1.0, places=0)
        self.assertTrue(type(gbest), Tensor)


if __name__ == "__main__":
    unittest.main()
