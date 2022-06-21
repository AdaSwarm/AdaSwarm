"""Rotated PSO algorithm."""
import time
from torch import Tensor
from torch.nn import CrossEntropyLoss
from adaswarm.particle import ParticleSwarm
from adaswarm.utils.options import get_device


class ParticleSwarmOptimizer:  # pylint: disable=R0902 R0913
    """Rotated Particle Swarm Optimizer"""

    def __init__(
        self,
        targets,
        dimension,
        number_of_classes,
        # TODO: associate swarm_size to dataset
        swarm_size,
        # TODO: associate accel coefficients to dataset
        acceleration_coefficients,
        # TODO: associate inertial weight to dataset
        inertial_weight_beta: float,
        max_iterations=100,
        device=get_device(),
    ):

        self.max_iterations = max_iterations
        self.gbest_position = None
        self.gbest_value = Tensor([float("inf")]).to(device)
        self.loss_function = CrossEntropyLoss()
        self.swarm_size = swarm_size
        self.device = device
        self.swarm = ParticleSwarm(
            dimension=dimension,
            number_of_classes=number_of_classes,
            swarm_size=swarm_size,
            acceleration_coefficients=acceleration_coefficients,
            inertial_weight_beta=inertial_weight_beta,
            targets=targets,
        )
        self.targets = targets

    def __run_one_iteration(self, verbosity=True):
        tic = time.monotonic()

        # --- Set PBest & GBest
        for particle in self.swarm:
            best_fitness_candidate = self.loss_function(
                particle.position, self.targets
            ).to(self.device)
            if particle.pbest_value > best_fitness_candidate:
                particle.pbest_value = best_fitness_candidate
                particle.pbest_position = particle.position.clone()
            if self.gbest_value > best_fitness_candidate:
                self.gbest_value = best_fitness_candidate
                self.gbest_position = particle.position.clone()

        self.swarm.update_velocities(self.gbest_position)

        toc = time.monotonic()
        if verbosity is True:
            print(
                f" >> global best fitness {self.gbest_value:.3f}  | iteration time {toc - tic:.3f}"
            )
        return self.gbest_position

    def run_iteration(self, number=1, verbosity=False):
        """Runs a number of iterations of the algorithm."""
        for _ in range(number):
            gbest = self.__run_one_iteration(verbosity=verbosity)
        return (self.swarm.average_of_scaled_acceleration_coefficients(), gbest)
