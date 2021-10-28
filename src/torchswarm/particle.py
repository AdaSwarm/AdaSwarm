import torch
import numpy as np
from torchswarm.utils.rpso import get_rotation_matrix, get_inverse_matrix, get_phi_matrix

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
        number_of_classes: int=1, 
        acceleration_coefficients: dict = {"c1":0.9, "c2":0.8},
        inertial_weight_w: float = 0.5,
        targets: list = []
        ):
        self.size = swarm_size
        self.initialise_with_particles(
            dimension=dimension, 
            swarm_size=swarm_size, 
            number_of_classes=number_of_classes,
            acceleration_coefficients=acceleration_coefficients,
            inertial_weight_beta=inertial_weight_w,
            targets=targets
            )

    def initialise_with_particles(
        self, 
        dimension: int, 
        swarm_size: int, 
        number_of_classes: int,
        acceleration_coefficients: dict,
        inertial_weight_beta: float,
        targets: any
        ):
        for _ in range(swarm_size):
            self.append(
                RotatedEMParticle(
                    dimension, 
                    inertial_weight_beta, 
                    acceleration_coefficients["c1"], 
                    acceleration_coefficients["c2"], 
                    number_of_classes
                    )
                )
    


class RotatedEMParticle(Particle):
    def __init__(self, dimensions, beta, c1, c2, classes, true_y):
        self.dimensions = dimensions
        self.position = torch.rand(dimensions, classes).to(device)
        self.velocity = torch.zeros((dimensions, classes)).to(device)
        self.pbest_value = torch.Tensor([float("inf")]).to(device)
        self.c1 = c1
        self.c2 = c2
        # print(to_categorical(true_y.cpu().detach().numpy()))
        self.position = initialize_position(true_y, dimensions, classes).to(device)
        self.pbest_position = self.position
        self.momentum = torch.zeros((dimensions, 1)).to(device)
        self.beta = beta

    def __str__(self):
        return ('Particle >> pbest {:.3f}  | pbest_position {}'
                .format(self.pbest_value.item(),self.pbest_position))


    def update_velocity(self, gbest_position):
        r1 = torch.rand(1)
        r2 = torch.rand(1)
        momentum_t = self.beta*self.momentum + (1 - self.beta)*self.velocity
        a_matrix = get_rotation_matrix(self.dimensions, np.pi/5, 0.4)
        a_inverse_matrix = get_inverse_matrix(a_matrix)
        # x = a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix 
        self.velocity = momentum_t \
                        + torch.matmul((a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix).float().to(device),(self.pbest_position - self.position).float().to(device)) \
                        + torch.matmul((a_inverse_matrix * get_phi_matrix(self.dimensions, self.c2, r2) * a_matrix).float().to(device), (gbest_position - self.position).float().to(device))

        return ((self.c1*r1).item(), (self.c2*r2).item())
    def move(self):
        for i in range(0, self.dimensions):
            # print("Before Update: ",self.position[i])
            self.position[i] = self.position[i] + self.velocity[i]
            # print("After Update: ",self.position[i], self.velocity[i])



class RotatedEMParticleWithBounds(Particle):
    def __init__(self, dimensions, beta, c1, c2, classes, bounds):
        super().__init__(dimensions, 0, c1, c2, classes)
        self.position = (bounds[0]-bounds[1])*torch.rand(dimensions, classes) + bounds[1]
        self.momentum = torch.zeros((dimensions, 1))
        self.beta = beta

    def update_velocity(self, gbest_position):
        r1 = torch.rand(1)
        r2 = torch.rand(1)
        momentum_t = self.beta*self.momentum + (1 - self.beta)*self.velocity
        a_matrix = get_rotation_matrix(self.dimensions, np.pi/5, 0.4)
        a_inverse_matrix = get_inverse_matrix(a_matrix)
        # x = a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix 
        self.velocity = momentum_t \
                        + torch.matmul((a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix).float().to(device),(self.pbest_position - self.position).float().to(device)) \
                        + torch.matmul((a_inverse_matrix * get_phi_matrix(self.dimensions, self.c2, r2) * a_matrix).float().to(device), (gbest_position - self.position).float().to(device))

        return ((self.c1*r1).item(), (self.c2*r2).item())
    def move(self):
        for i in range(0, self.dimensions):
            # print("Before Update: ",self.position[i])
            self.position[i] = self.position[i] + self.velocity[i]
            # print("After Update: ",self.position[i], self.velocity[i])
        self.position = torch.clamp(self.position,-50,50)
        

def initialize_position(true_y, dimensions, classes):
    const = -4
    position = torch.tensor([[const]*classes]*dimensions)
    for i in range(dimensions):
        position[i][true_y[i]] = 1
    return position + torch.rand(dimensions, classes)