
import time
from torch import device as torch_device, cuda, Tensor
from torch.nn import CrossEntropyLoss
from torchswarm.particle import ParticleSwarm


class RotatedEMParticleSwarmOptimizer:
    def __init__(
        self,
        dimension=4,
        swarm_size=100,
        number_of_classes=1,
        targets=None,
        options=None,
        device=torch_device("cuda:0" if cuda.is_available() else "cpu")
    ):
        # TODO: This is unclear pass named options
        if options == None:
            options = [2, 2, 0.1, 100]
        self.max_iterations = options[3]
        self.targets = targets
        self.gbest_position = None
        self.gbest_value = Tensor([float("inf")]).to(device)
        self.loss_function = CrossEntropyLoss()
        self.swarm_size = swarm_size
        self.device = device
        self.swarm = ParticleSwarm(
            dimension=dimension,
            number_of_classes=number_of_classes,
            swarm_size=swarm_size,
            acceleration_coefficients={"c1": options[0], "c2": options[1]},
            inertial_weight_beta=options[2],
            targets=targets,
        )

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

    def run_one_iter(self, verbosity=True):
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
