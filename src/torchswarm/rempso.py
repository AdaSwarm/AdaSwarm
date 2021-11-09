import time
from torch import device as torch_device, cuda, Tensor, randint
from torch.nn import CrossEntropyLoss
from torchswarm.particle import ParticleSwarm


class RotatedEMParticleSwarmOptimizer:
    def __init__(
        self,
        targets,
        dimension,
        number_of_classes,
        swarm_size=100,
        acceleration_coefficients: dict = {"c1": 2, "c2": 2},
        inertial_weight_beta: float = 0.1,
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

    def run(self, verbosity=True):
        # --- Run
        for iteration in range(self.max_iterations):
            tic = time.monotonic()
            # --- Set PBest
            for particle in self.swarm:
                fitness_candidate = self.loss_function(particle.position, self.targets)
                # print("========: ", fitness_candidate, particle.pbest_value)
                if particle.pbest_value > fitness_candidate:
                    particle.pbest_value = fitness_candidate
                    particle.pbest_position = particle.position.clone()
                # print("========: ",particle.pbest_value)
            # --- Set GBest
            for particle in self.swarm:
                best_fitness_candidate = self.loss_function(
                    particle.position, self.targets
                )
                if self.gbest_value > best_fitness_candidate:
                    self.gbest_value = best_fitness_candidate
                    self.gbest_position = particle.position.clone()

            # --- For Each Particle Update Velocity
            for particle in self.swarm:
                particle.update_velocity(self.gbest_position)
                particle.move()
            # for particle in self.swarm:
            #     print(particle)
            # print(self.gbest_position.numpy())
            toc = time.monotonic()
            if verbosity == True:
                print(
                    "Iteration {:.0f} >> global best fitness {:.3f}  | iteration time {:.3f}".format(
                        iteration + 1, self.gbest_value, toc - tic
                    )
                )
            if iteration + 1 == self.max_iterations:
                print(self.gbest_position)

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

        c1r1s = []
        c2r2s = []
        # --- For Each Particle Update Velocity
        for particle in self.swarm:
            c1r1, c2r2 = particle.update_velocity(self.gbest_position)
            particle.move()
            c1r1s.append(c1r1)
            c2r2s.append(c2r2)
        # for particle in self.swarm:
        #     print(particle)
        # print(self.gbest_position.numpy())
        toc = time.monotonic()
        if verbosity == True:
            print(
                " >> global best fitness {:.3f}  | iteration time {:.3f}".format(
                    self.gbest_value, toc - tic
                )
            )
        return (
            sum(c1r1s) / self.swarm_size,
            sum(c2r2s) / self.swarm_size,
            self.gbest_position,
        )

    def run_iteration(self, number=1, verbosity=False):
        c1r1 = c2r2 = gbest = 0.0
        for _ in range(number):
            c1r1, c2r2, gbest = self.__run_one_iteration(verbosity=verbosity)
        return (c1r1, c2r2, gbest)
