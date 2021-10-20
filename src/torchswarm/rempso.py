import torch 
import time
from torchswarm.particle import RotatedEMParticle, RotatedEMParticleWithBounds
from torch.nn import CrossEntropyLoss

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev) 

class RotatedEMParticleSwarmOptimizer:
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
        self.loss_function = CrossEntropyLoss()

        for _ in range(swarm_size):
            self.swarm.append(RotatedEMParticle(dimensions, self.beta, self.c1, self.c2, classes, self.true_y))
    

    def run(self,verbosity = True):
        #--- Run 
        for iteration in range(self.max_iterations):
            tic = time.monotonic()
            #--- Set PBest
            for particle in self.swarm:
                fitness_candidate = self.loss_function.fitness(particle.position, self.true_y)
                # print("========: ", fitness_cadidate, particle.pbest_value)
                if(particle.pbest_value > fitness_candidate):
                    particle.pbest_value = fitness_candidate
                    particle.pbest_position = particle.position.clone()
                # print("========: ",particle.pbest_value)
            #--- Set GBest
            for particle in self.swarm:
                best_fitness_cadidate = self.loss_function.evaluate(particle.position)
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
            fitness_cadidate = self.loss_function.evaluate(particle.position).to(device)
            # print("========: ", fitness_cadidate, particle.pbest_value)
            if(particle.pbest_value > fitness_cadidate):
                particle.pbest_value = fitness_cadidate
                particle.pbest_position = particle.position.clone()
            # print("========: ",particle.pbest_value)
        #--- Set GBest
        for particle in self.swarm:
            best_fitness_cadidate = self.loss_function.evaluate(particle.position).to(device)
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

class RotatedEMParticleSwarmOptimizerWithBounds:
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

        for _ in range(swarm_size):
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