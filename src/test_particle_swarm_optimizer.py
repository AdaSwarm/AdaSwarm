import unittest
from torch import randint, device as torch_device
from torch.autograd import Variable
from torchswarm.rempso import RotatedEMParticleSwarmOptimizer


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
            targets=targets, 
            dimension=dimension, 
            number_of_classes=number_of_classes
        )
        
        c1r1, c2r2, gbest = optimizer.run_iteration(number=5)
        self.assertEqual(c1r1, 2)
        self.assertEqual(c2r2, 2)
        self.assertEqual(gbest, 2)


if __name__ == "__main__":
    unittest.main()
