import torch
import numpy as np
from torch._C import dtype
from torchswarm.utils.rpso import (
    get_rotation_matrix,
    get_inverse_matrix,
    get_phi_matrix,
)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


class ParticleSwarm(list):
    def __init__(
        self,
        dimension: int = 125,
        swarm_size: int = 100,
        number_of_classes: int = 1,
        acceleration_coefficients: dict = {"c1": 0.9, "c2": 0.8},
        inertial_weight_beta: float = 0.5,
        targets: torch.Tensor = torch.empty([125], dtype=torch.long),
    ):
        self.size = swarm_size
        for _ in range(swarm_size):
            self.append(
                RotatedEMParticle(
                    dimension,
                    inertial_weight_beta,
                    acceleration_coefficients["c1"],
                    acceleration_coefficients["c2"],
                    number_of_classes,
                    targets,
                )
            )


class RotatedEMParticle:
    def __init__(self, dimensions, beta, c1, c2, number_of_classes, targets):
        self.dimensions = dimensions
        self.velocity = torch.zeros((dimensions, number_of_classes)).to(device)
        self.pbest_value = torch.Tensor([float("inf")]).to(device)
        self.c1 = c1
        self.c2 = c2
        self.position = self.initialize_position(
            targets, dimensions, number_of_classes
        ).to(device)
        self.pbest_position = self.position
        self.momentum = torch.zeros((dimensions, 1)).to(device)
        self.beta = beta

    def __str__(self):
        return "Particle >> pbest {:.3f}  | pbest_position {}".format(
            self.pbest_value.item(), self.pbest_position
        )

    def update_velocity(self, gbest_position):
        r1 = torch.rand(1)
        r2 = torch.rand(1)
        momentum_t = self.beta * self.momentum + (1 - self.beta) * self.velocity
        a_matrix = get_rotation_matrix(self.dimensions, np.pi / 5, 0.4)
        a_inverse_matrix = get_inverse_matrix(a_matrix)
        # x = a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix
        self.velocity = (
            momentum_t
            + torch.matmul(
                (
                    a_inverse_matrix
                    * get_phi_matrix(self.dimensions, self.c1, r1)
                    * a_matrix
                )
                .float()
                .to(device),
                (self.pbest_position - self.position).float().to(device),
            )
            + torch.matmul(
                (
                    a_inverse_matrix
                    * get_phi_matrix(self.dimensions, self.c2, r2)
                    * a_matrix
                )
                .float()
                .to(device),
                (gbest_position - self.position).float().to(device),
            )
        )

        return ((self.c1 * r1).item(), (self.c2 * r2).item())

    def move(self):
        for i in range(0, self.dimensions):
            # print("Before Update: ",self.position[i])
            self.position[i] = self.position[i] + self.velocity[i]
            # print("After Update: ",self.position[i], self.velocity[i])

    def initialize_position(self, targets, dimensions, number_of_classes):
        const = -4
        position = torch.tensor([[const] * number_of_classes] * dimensions)
        for i in range(dimensions):
            position[i][targets[i]] = 1
        return position + torch.rand(dimensions, number_of_classes)
