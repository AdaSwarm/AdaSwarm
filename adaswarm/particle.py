"""
Particles and swarms thereof; to be used by the Particle Swarm Optimizer (PSO)

PSO navigates the search space by embracing a swarm which is
nothing but a population of particles. The swarm, guided by characteristic
equations, attempt to converge to an optima [Ebenhart and Shi, 1995].
"""

import numpy as np
import torch

from adaswarm.utils.matrix import get_phi_matrix, get_rotation_matrix
from adaswarm.utils.options import dataset_name, get_device


class AccelerationCoefficients:
    """Acceleration coefficients for PSO"""

    # TODO: Define a framework for setting all these paramters
    def __init__(self):
        self.c_1 = 0.2 if dataset_name() == "MNIST" else 0.9
        self.c_2 = 0.8

    def sum(self):
        """Sum of the acceleration coefficients"""
        return self.c_1 + self.c_2

    def random_scale_c_1(self):
        """Randomly scale the acceleration coefficients"""
        return self.c_1 * torch.rand(1)

    def random_scale_c_2(self):
        """Randomly scale the acceleration coefficients"""
        return self.c_2 * torch.rand(1)


class RotatedEMParticle:
    """Exponentially weighted Momentum Particle"""

    # pylint: disable=R0902, R0913

    def __init__(
        self,
        dimensions,
        beta,
        acceleration_coefficients,
        number_of_classes,
        targets,
        device,
    ):
        self.device = device
        self.dimensions = dimensions
        self.velocity = torch.zeros((dimensions, number_of_classes)).to(device)
        self.pbest_value = torch.Tensor([float("inf")]).to(device)
        self.acceleration_coefficients = acceleration_coefficients
        self.position = _initialize_position(
            targets=targets, dimensions=dimensions, number_of_classes=number_of_classes
        ).to(device)
        self.pbest_position = self.position
        self.momentum = torch.zeros((dimensions, 1)).to(device)
        self.beta = beta
        self.targets = targets
        self.c_1_r_1 = 0.0
        self.c_2_r_2 = 0.0

    def __str__(self):
        return f"Particle >> pbest {self.pbest_value.item():.3f}  | \
            pbest_position {self.pbest_position}"

    def update_velocity(self, gbest_position):
        """Velocity is the mechanism used to move (evolve) the position of
        a particle to search for optimal solutions.

        Args:
            gbest_position (float): global best position of the swarm

        Returns:
            [Tuple]: Updated acceleration coefficients
        """
        scaled_c_1_tensor = self.acceleration_coefficients.random_scale_c_1()
        scaled_c_2_tensor = self.acceleration_coefficients.random_scale_c_2()
        momentum_t = self.beta * self.momentum + (1 - self.beta) * self.velocity
        a_matrix = get_rotation_matrix(self.dimensions, np.pi / 5, 0.4)
        a_inverse_matrix = torch.inverse(a_matrix)
        # TODO: check paper
        # TODO: x = a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix
        self.velocity = (
            momentum_t
            + torch.matmul(
                (
                    a_inverse_matrix
                    * get_phi_matrix(self.dimensions, scaled_c_1_tensor)
                    * a_matrix
                )
                .float()
                .to(self.device),
                (self.pbest_position - self.position).float().to(self.device),
            )
            + torch.matmul(
                (
                    a_inverse_matrix
                    * get_phi_matrix(self.dimensions, scaled_c_2_tensor)
                    * a_matrix
                )
                .float()
                .to(self.device),
                (gbest_position - self.position).float().to(self.device),
            )
        )
        self.move()
        self.c_1_r_1 = scaled_c_1_tensor.item()
        self.c_2_r_2 = scaled_c_2_tensor.item()

    def move(self):
        """This evolves the position of the particle by the amount set in the velocity"""
        self.position += self.velocity


class ParticleSwarm(list):
    """Wrapper for a collection of particles"""

    # pylint: disable=R0913

    def __init__(
        self,
        targets,
        dimension,
        number_of_classes,
        swarm_size,
        acceleration_coefficients: AccelerationCoefficients,
        inertial_weight_beta: float,
        device: str = get_device(),
    ):
        super().__init__(self)
        self.size = swarm_size

        for _ in range(swarm_size):
            self.append(
                RotatedEMParticle(
                    dimensions=dimension,
                    beta=inertial_weight_beta,
                    acceleration_coefficients=acceleration_coefficients,
                    number_of_classes=number_of_classes,
                    targets=targets,
                    device=device,
                )
            )

    def update_velocities(self, gbest_position):
        """Compute new velocities to enable calculation of the next position of
        each particle.

        Args:
            gbest_position (torch.Tensor): Input tensor containing the global best
            known value of all particles of the swarm.

        Returns:
            [list of floats, list of floats]: Two lists containing the c1r1 and c2r2
            float values for the entire swarm (these are acceleration coefficients c1
            and c2 scaled by a random number; r1 and r2 respectively).
        """

        def update_velocity(particle):
            """Velocity is the mechanism used to move (evolve) the position of
            a particle to search for optimal solutions.

            Args:
                gbest_position (float): global best position of the swarm

            Returns:
                [Tuple]: Updated acceleration coefficients
            """
            scaled_c_1_tensor = particle.acceleration_coefficients.random_scale_c_1()
            scaled_c_2_tensor = particle.acceleration_coefficients.random_scale_c_2()
            momentum_t = (
                particle.beta * particle.momentum
                + (1 - particle.beta) * particle.velocity
            )
            a_matrix = get_rotation_matrix(particle.dimensions, np.pi / 5, 0.4)
            a_inverse_matrix = torch.inverse(a_matrix)
            # TODO: check paper
            # TODO: x = a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix
            particle.velocity = (
                momentum_t
                + torch.matmul(
                    (
                        a_inverse_matrix
                        * get_phi_matrix(particle.dimensions, scaled_c_1_tensor)
                        * a_matrix
                    )
                    .float()
                    .to(particle.device),
                    (particle.pbest_position - particle.position)
                    .float()
                    .to(particle.device),
                )
                + torch.matmul(
                    (
                        a_inverse_matrix
                        * get_phi_matrix(particle.dimensions, scaled_c_2_tensor)
                        * a_matrix
                    )
                    .float()
                    .to(particle.device),
                    (gbest_position - particle.position).float().to(particle.device),
                )
            )
            particle.move()
            particle.c_1_r_1 = scaled_c_1_tensor.item()
            particle.c_2_r_2 = scaled_c_2_tensor.item()

        # TODO: Vectorize this operation
        for particle in self:
            update_velocity(particle)

    def average_of_scaled_acceleration_coefficients(self):
        """Compute average of scaled coefficients"""
        return sum((particle.c_1_r_1 + particle.c_2_r_2) for particle in self) / len(
            self
        )


def _initialize_position(targets, dimensions, number_of_classes):
    #TODO: Not hardcoded ideally
    if dataset_name() in ["Iris"]:
        return torch.clamp(
            input=targets
            + (0.1 ** 0.5)
            * torch.rand(dimensions, number_of_classes, device=get_device()),
            min=0,
            max=1,
        )
    const = -4
    position = torch.tensor([[const] * number_of_classes] * dimensions)
    for i in range(dimensions):
        position[i][targets[i]] = 1
    return position + torch.rand(dimensions, number_of_classes)
