import torch
import numpy as np
from torchswarm.utils.rpso import get_rotation_matrix, get_inverse_matrix, get_phi_matrix

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev) 

class Particle:
    def __init__(self, dimensions, w, c1, c2, classes):
        self.dimensions = dimensions
        self.position = torch.rand(dimensions, classes).to(device)
        self.velocity = torch.zeros((dimensions, classes)).to(device)
        self.pbest_position = self.position
        self.pbest_value = torch.Tensor([float("inf")]).to(device)
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def __str__(self):
        return ('Particle >> pbest {:.3f}  | pbest_position {}'
                .format(self.pbest_value.item(),self.pbest_position))
        
    def update_velocity(self, gbest_position):
        r1 = torch.rand(1)
        r2 = torch.rand(1)
        for i in range(0, self.dimensions):
            # print(self.velocity[i], (self.pbest_position[i]), .(gbest_position[i] - self.position[i]))
            self.velocity[i] = self.w * self.velocity[i] \
                                + self.c1 * r1 * (self.pbest_position[i] - self.position[i]) \
                                + self.c2 * r2 * (gbest_position[i] - self.position[i])

            # print(self.velocity[i])
        return ((self.c1*r1).item(), (self.c2*r2).item())
    
    def move(self):
        for i in range(0, self.dimensions):
            # print("Before Update: ",self.position[i])
            self.position[i] = self.position[i] + self.velocity[i]
            # print("After Update: ",self.position[i], self.velocity[i])
        self.position = torch.clamp(self.position,0,1)

class RotatedParticle(Particle):
    def update_velocity(self, gbest_position):
        r1 = torch.rand(1)
        r2 = torch.rand(1)
        a_matrix = get_rotation_matrix(self.dimensions, np.pi/5, 0.4)
        a_inverse_matrix = get_inverse_matrix(a_matrix)
        # x = a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix 
        self.velocity = self.w * self.velocity \
                        + torch.matmul((a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix).float().to(device),(self.pbest_position - self.position).float().to(device)) \
                        + torch.matmul((a_inverse_matrix * get_phi_matrix(self.dimensions, self.c2, r2) * a_matrix).float().to(device), (gbest_position - self.position).float().to(device))
        return ((self.c1*r1).item(), (self.c2*r2).item())

class EMParticle(Particle):
    def __init__(self, dimensions, beta, c1, c2, classes):
        super().__init__(dimensions, 0, c1, c2, classes)
        self.momentum = torch.zeros((dimensions, 1))
        self.beta = beta

    def update_velocity(self, gbest_position):
        r1 = torch.rand(1)
        r2 = torch.rand(1)
        for i in range(0, self.dimensions):
            # print(self.velocity[i], (self.pbest_position[i]), (gbest_position[i] - self.position[i]))
            momentum_t = self.beta*self.momentum[i] + (1 - self.beta)*self.velocity[i]
            self.velocity[i] = momentum_t \
                                + self.c1 * r1 * (self.pbest_position[i] - self.position[i]) \
                                + self.c2 * r2 * (gbest_position[i] - self.position[i])
            self.momentum[i] = momentum_t

            # print(self.velocity[i])
        return ((self.c1*r1).item(), (self.c2*r2).item())

class RotatedEMParticle(Particle):
    def __init__(self, dimensions, beta, c1, c2, classes, true_y):
        super().__init__(dimensions, 0, c1, c2, classes)
        # print(to_categorical(true_y.cpu().detach().numpy()))
        self.position = initialize_position(true_y, dimensions, classes).to(device)
        self.pbest_position = self.position
        self.momentum = torch.zeros((dimensions, 1)).to(device)
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