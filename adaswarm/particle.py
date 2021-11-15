"""
Particles and swarms thereof; to be used by the Particle Swarm Optimizer (PSO)

PSO navigates the search space by embracing a swarm which is
nothing but a population of particles. The swarm, guided by characteristic
equations, attempt to converge to an optima [Ebenhart and Shi, 1995].
"""

import torch
import numpy as np
from adaswarm.utils.matrix import (
    get_rotation_matrix,
    get_inverse_matrix,
    get_phi_matrix,
)


class AccelerationCoefficients:
    """Acceleration coefficients for PSO"""

    def __init__(self, c_1: float = 0.9, c_2: float = 0.8):
        self.c_1 = c_1
        self.c_2 = c_2

    def sum(self):
        """Sum of the acceleration coefficients"""
        return self.c_1 + self.c_2


class ParticleSwarm(list):
    """Wrapper for a collection of particles"""

    # pylint: disable=R0913

    def __init__(
        self,
        targets,
        dimension,
        number_of_classes,
        swarm_size: int = 100,
        acceleration_coefficients: AccelerationCoefficients = AccelerationCoefficients(),
        inertial_weight_beta: float = 0.5,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
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
        r_1 = torch.rand(1)
        r_2 = torch.rand(1)
        momentum_t = self.beta * self.momentum + \
            (1 - self.beta) * self.velocity
        a_matrix = get_rotation_matrix(self.dimensions, np.pi / 5, 0.4)
        a_inverse_matrix = get_inverse_matrix(a_matrix)
        # TODO: check paper
        # TODO: x = a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix
        self.velocity = (
            momentum_t
            + torch.matmul(
                (
                    a_inverse_matrix
                    * get_phi_matrix(
                        self.dimensions, self.acceleration_coefficients.c_1, r_1
                    )
                    * a_matrix
                )
                .float()
                .to(self.device),
                (self.pbest_position - self.position).float().to(self.device),
            )
            + torch.matmul(
                (
                    a_inverse_matrix
                    * get_phi_matrix(
                        self.dimensions, self.acceleration_coefficients.c_2, r_2
                    )
                    * a_matrix
                )
                .float()
                .to(self.device),
                (gbest_position - self.position).float().to(self.device),
            )
        )

        return (
            (self.acceleration_coefficients.c_1 * r_1).item(),
            (self.acceleration_coefficients.c_2 * r_2).item(),
        )

    def move(self):
        """This evolves the position of the particle by the amount set in the velocity
        """
        # TODO: tidy up loop and use of indexes
        for i in range(0, self.dimensions):
            # print("Before Update: ",self.position[i])
            self.position[i] = self.position[i] + self.velocity[i]
            # print("After Update: ",self.position[i], self.velocity[i])


def _initialize_position(targets, dimensions, number_of_classes):
    const = -4
    position = torch.tensor([[const] * number_of_classes] * dimensions)
    for i in range(dimensions):
        position[i][targets[i]] = 1
    return position + torch.rand(dimensions, number_of_classes)
