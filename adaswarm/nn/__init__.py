import torch
import torch.nn.functional as F
from adaswarm.rempso import ParticleSwarmOptimizer
from adaswarm.particle import AccelerationCoefficients


def CrossEntropyLoss():
    return CrossEntropyLossCreator.CrossEntropyLossWithParticleSwarmOptimizer.apply


def BCELoss():
    return (
        CrossEntropyLossCreator.BinaryCrossEntropyLossWithParticleSwarmOptimizer.apply
    )


class CrossEntropyLossCreator:
    @staticmethod
    def apply_forward(
        ctx, y, y_pred, swarm_learning_rate, entropy_loss=F.cross_entropy
    ):
        dimension, classes = y.size()
        # TODO: Check the optimum swarm size
        acceleration_coefficients=AccelerationCoefficients()
        inertial_weight_beta =  0.1
        swarm_size = 10

        particle_swarm_optimizer = ParticleSwarmOptimizer(
            dimension=dimension,
            swarm_size=swarm_size,
            acceleration_coefficients=acceleration_coefficients,
            inertial_weight_beta=inertial_weight_beta,
            number_of_classes=classes,
            targets=y_pred,
        )
        sum_cr, gbest = particle_swarm_optimizer.run_iteration(number=40)
        ctx.save_for_backward(y, y_pred)
        ctx.sum_cr = sum_cr
        ctx.eta = swarm_learning_rate
        ctx.gbest = gbest
        return entropy_loss(y, y_pred)

    @staticmethod
    def apply_backward(ctx, grad_output):
        yy, _ = ctx.saved_tensors
        sum_cr = ctx.sum_cr
        eta = ctx.eta
        grad_input = torch.neg((sum_cr / eta) * (ctx.gbest - yy))
        return grad_input, grad_output, None, None, None

    class CrossEntropyLossWithParticleSwarmOptimizer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y, y_pred, swarm_learning_rate=0.1):
            return CrossEntropyLossCreator.apply_forward(ctx, y, y_pred, swarm_learning_rate)

        @staticmethod
        def backward(ctx, grad_output):
            return CrossEntropyLossCreator.apply_backward(ctx, grad_output)

    class BinaryCrossEntropyLossWithParticleSwarmOptimizer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y, y_pred, swarm_learning_rate=0.1):
            return CrossEntropyLossCreator.apply_forward(
                ctx, y, y_pred, swarm_learning_rate, F.binary_cross_entropy
            )

        @staticmethod
        def backward(ctx, grad_output):
            return CrossEntropyLossCreator.apply_backward(ctx, grad_output)
