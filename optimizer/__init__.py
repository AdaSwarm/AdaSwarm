import torch
import torch.nn.functional as F
import torch.optim as optim
from .optimizer import Optimizer
import torch 
import time
from torchswarm_gpu.particle import RotatedEMParticle, RotatedEMParticleWithBounds
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev) 


class AdaSwarm(Optimizer):
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        self.adam_optimizer = optim.Adam(params, lr)
        self.particle_swarm_optimizer = RotatedEMParicleSwarmOptimizer(batch_size, swarm_size, 10, targets)
        self.approx_criterion = CELossWithPSO.apply
        

    def step(self, closure=None):
        self.particle_swarm_optimizer.optimize(CELoss(targets))
        for i in range(5):
            c1r1, c2r2, gbest = self.particle_swarm_optimizerp.run_one_iter(verbosity=False)
        optimizer.zero_grad()
        # training code should setup net(neural net architecture) 
        gbest = torch.clamp(torch.exp(gbest), 0, 1)
        loss = self.approx_criterion(outputs, targets, c1r1+c2r2, 5, gbest)
        loss.backward()
        self.adam_optimizer.step()

class CELoss:
    def __init__(self, y):
        self.y = y
        self.fitness = torch.nn.CrossEntropyLoss()
    def evaluate(self, x):
        # print(x, self.y)
        return self.fitness(x, self.y)

class CELossWithPSO(torch.autograd.Function):  
    @staticmethod
    def forward(ctx , y, y_pred, sum_cr, eta, gbest):
        ctx.save_for_backward(y, y_pred)
        ctx.sum_cr = sum_cr
        ctx.eta = eta
        ctx.gbest = gbest
        return F.cross_entropy(y,y_pred)

    @staticmethod
    def backward(ctx, grad_output):
        yy, yy_pred= ctx.saved_tensors
        sum_cr = ctx.sum_cr
        eta = ctx.eta
        grad_input = torch.neg((sum_cr/eta) * (ctx.gbest - yy))
        return grad_input, grad_output, None, None, None


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
        x = a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix 
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
        x = a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix 
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
        x = a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix 
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

class RotatedEMParicleSwarmOptimizer:
    def __init__(self,dimensions = 4, swarm_size=100,classes=1, true_y=None, options=None):
        if (options == None):
            options = [2,2,0.1,100]
        self.swarm_size = swarm_size
        self.c1 = options[0]
        self.c2 = options[1]
        self.beta = options[2]
        self.max_iterations = options[3]
        self.swarm = []
        self.true_y = true_y
        self.gbest_position = None
        self.gbest_value = torch.Tensor([float("inf")]).to(device)

        for i in range(swarm_size):
            self.swarm.append(RotatedEMParticle(dimensions, self.beta, self.c1, self.c2, classes, self.true_y))
    
    def optimize(self, function):
        self.fitness_function = function

    def run(self,verbosity = True):
        #--- Run 
        for iteration in range(self.max_iterations):
            tic = time.monotonic()
            #--- Set PBest
            for particle in self.swarm:
                fitness_cadidate = self.fitness_function.evaluate(particle.position)
                # print("========: ", fitness_cadidate, particle.pbest_value)
                if(particle.pbest_value > fitness_cadidate):
                    particle.pbest_value = fitness_cadidate
                    particle.pbest_position = particle.position.clone()
                # print("========: ",particle.pbest_value)
            #--- Set GBest
            for particle in self.swarm:
                best_fitness_cadidate = self.fitness_function.evaluate(particle.position)
                if(self.gbest_value > best_fitness_cadidate):
                    self.gbest_value = best_fitness_cadidate
                    self.gbest_position = particle.position.clone()

            #--- For Each Particle Update Velocity
            for particle in self.swarm:
                particle.update_velocity(self.gbest_position)
                particle.move()
            # for particle in self.swarm:
            #     print(particle)
            # print(self.gbest_position.numpy())
            toc = time.monotonic()
            if (verbosity == True):
                print('Iteration {:.0f} >> global best fitness {:.3f}  | iteration time {:.3f}'
                .format(iteration + 1,self.gbest_value,toc-tic))
            if(iteration+1 == self.max_iterations):
                print(self.gbest_position)

    def run_one_iter(self, verbosity=True):
        tic = time.monotonic()
        #--- Set PBest
        for particle in self.swarm:
            fitness_cadidate = self.fitness_function.evaluate(particle.position).to(device)
            # print("========: ", fitness_cadidate, particle.pbest_value)
            if(particle.pbest_value > fitness_cadidate):
                particle.pbest_value = fitness_cadidate
                particle.pbest_position = particle.position.clone()
            # print("========: ",particle.pbest_value)
        #--- Set GBest
        for particle in self.swarm:
            best_fitness_cadidate = self.fitness_function.evaluate(particle.position).to(device)
            if(self.gbest_value > best_fitness_cadidate):
                self.gbest_value = best_fitness_cadidate
                self.gbest_position = particle.position.clone()

        c1r1s = []
        c2r2s = []
        #--- For Each Particle Update Velocity
        for particle in self.swarm:
            c1r1, c2r2 = particle.update_velocity(self.gbest_position)
            particle.move()
            c1r1s.append(c1r1)
            c2r2s.append(c2r2)
        # for particle in self.swarm:
        #     print(particle)
        # print(self.gbest_position.numpy())
        toc = time.monotonic()
        if (verbosity == True):
            print(' >> global best fitness {:.3f}  | iteration time {:.3f}'
            .format(self.gbest_value,toc-tic))
        return (sum(c1r1s)/ self.swarm_size, sum(c2r2s)/ self.swarm_size, self.gbest_position)

class RotatedEMParicleSwarmOptimizerWithBounds:
    def __init__(self,dimensions = 4, swarm_size=100,classes=1, bounds=None, options=None):
        if (options == None):
            options = [0.9,0.8,0.1,100]
        self.swarm_size = swarm_size
        self.c1 = options[0]
        self.c2 = options[1]
        self.beta = options[2]
        self.max_iterations = options[3]
        self.swarm = []
        self.gbest_position = None
        self.gbest_value = torch.Tensor([float("inf")])

        for i in range(swarm_size):
            self.swarm.append(RotatedEMParticleWithBounds(dimensions, self.beta, self.c1, self.c2, classes, bounds))
    
    def optimize(self, function):
        self.fitness_function = function

    def run(self,verbosity = True):
        #--- Run 
        for iteration in range(self.max_iterations):
            tic = time.monotonic()
            #--- Set PBest
            for particle in self.swarm:
                fitness_cadidate = self.fitness_function.evaluate(particle.position)
                # print("========: ", fitness_cadidate, particle.pbest_value)
                if(particle.pbest_value > fitness_cadidate):
                    particle.pbest_value = fitness_cadidate
                    particle.pbest_position = particle.position.clone()
                # print("========: ",particle.pbest_value)
            #--- Set GBest
            for particle in self.swarm:
                best_fitness_cadidate = self.fitness_function.evaluate(particle.position)
                if(self.gbest_value > best_fitness_cadidate):
                    self.gbest_value = best_fitness_cadidate
                    self.gbest_position = particle.position.clone()

            #--- For Each Particle Update Velocity
            for particle in self.swarm:
                particle.update_velocity(self.gbest_position)
                particle.move()
            # for particle in self.swarm:
            #     print(particle)
            # print(self.gbest_position.numpy())
            print(self.gbest_value)
            toc = time.monotonic()
            if (verbosity == True):
                print('Iteration {:.0f} >> global best fitness {:.3f}  | iteration time {:.3f}'
                .format(iteration + 1,self.gbest_value,toc-tic))
            if(iteration+1 == self.max_iterations):
                print(self.gbest_position)

    def run_one_iter(self, verbosity=True):
        tic = time.monotonic()
        #--- Set PBest
        for particle in self.swarm:
            fitness_cadidate = self.fitness_function.evaluate(particle.position)
            # print("========: ", fitness_cadidate, particle.pbest_value)
            if(particle.pbest_value > fitness_cadidate):
                particle.pbest_value = fitness_cadidate
                particle.pbest_position = particle.position.clone()
            # print("========: ",particle.pbest_value)
        #--- Set GBest
        for particle in self.swarm:
            best_fitness_cadidate = self.fitness_function.evaluate(particle.position)
            if(self.gbest_value > best_fitness_cadidate):
                self.gbest_value = best_fitness_cadidate
                self.gbest_position = particle.position.clone()

        c1r1s = []
        c2r2s = []
        #--- For Each Particle Update Velocity
        for particle in self.swarm:
            c1r1, c2r2 = particle.update_velocity(self.gbest_position)
            particle.move()
            c1r1s.append(c1r1)
            c2r2s.append(c2r2)
        # for particle in self.swarm:
        #     print(particle)
        # print(self.gbest_position.numpy())
        toc = time.monotonic()
        if (verbosity == True):
            print(' >> global best fitness {:.3f}  | iteration time {:.3f}'
            .format(self.gbest_value.item(),toc-tic))
        return (sum(c1r1s)/ self.swarm_size, sum(c2r2s)/ self.swarm_size, self.gbest_position)