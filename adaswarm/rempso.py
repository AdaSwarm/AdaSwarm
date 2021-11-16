import time
from torch import device as torch_device, cuda, Tensor, randint
from torch.nn import CrossEntropyLoss
from adaswarm.particle import ParticleSwarm, AccelerationCoefficients
import logging


class RotatedEMParticleSwarmOptimizer:
    def __init__(
        self,
        targets,
        dimension,
        number_of_classes,
        swarm_size=100,
        acceleration_coefficients=AccelerationCoefficients(c_1=0.2, c_2=0.8),
        inertial_weight_beta: float = 0.9,
        max_iterations=100,
        device=torch_device("cuda:0" if cuda.is_available() else "cpu"),
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
        # --- Set PBest
        for particle in self.swarm:
            fitness_candidate = self.loss_function(particle.position, self.targets).to(
                self.device
            )
            # print("========: ", fitness_candidate, particle.pbest_value)
            if particle.pbest_value > fitness_candidate:
                particle.pbest_value = fitness_candidate
                particle.pbest_position = particle.position.clone()
            # print("========: ",particle.pbest_value)
        # --- Set GBest
        for particle in self.swarm:
            best_fitness_candidate = self.loss_function(
                particle.position, self.targets
            ).to(self.device)
            if self.gbest_value > best_fitness_candidate:
                self.gbest_value = best_fitness_candidate
                self.gbest_position = particle.position.clone()

        # TODO: use acceleration coefficient object
        c1r1_list, c2r2_list = self.swarm.update_velocities(
            self.gbest_position)
        # TODO: use acceleration coefficient class object instead of list

        toc = time.monotonic()
        if verbosity is True:
            print(
                " >> global best fitness {:.3f}  | iteration time {:.3f}".format(
                    self.gbest_value, toc - tic
                )
            )
        return (
            sum(c1r1_list) / self.swarm_size,
            sum(c2r2_list) / self.swarm_size,
            self.gbest_position,
        )

    def run_iteration(self, number=1, verbosity=False):
        average_c1r1 = average_c2r2 = gbest = 0.0
        for _ in range(number):
            average_c1r1, average_c2r2, gbest = self.__run_one_iteration(verbosity=verbosity)
        return (average_c1r1, average_c2r2, gbest)
