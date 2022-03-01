import torch
import torch.nn.functional as F
from adaswarm.rempso import ParticleSwarmOptimizer


class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, y_pred, swarm_learning_rate=0.1):
        dimension, classes = y.size()
        # TODO: Check the optimum swarm size
        particle_swarm_optimizer = ParticleSwarmOptimizer(
            dimension=dimension, swarm_size=10, number_of_classes=classes, targets=y_pred
        )
        sum_cr, gbest = particle_swarm_optimizer.run_iteration(number=5)
        ctx.save_for_backward(y, y_pred)
        ctx.sum_cr = sum_cr
        ctx.eta = swarm_learning_rate
        ctx.gbest = gbest
        return F.cross_entropy(y, y_pred)

    @staticmethod
    def backward(ctx, grad_output):
        yy, _ = ctx.saved_tensors
        sum_cr = ctx.sum_cr
        eta = ctx.eta
        grad_input = torch.neg((sum_cr / eta) * (ctx.gbest - yy))
        return grad_input, grad_output, None, None, None

