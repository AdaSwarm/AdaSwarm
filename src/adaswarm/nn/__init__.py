"""
Utilities for creating the loss function
"""
import torch
import torch.nn.functional as F
from adaswarm.rempso import ParticleSwarmOptimizer
from adaswarm.particle import AccelerationCoefficients


def CrossEntropyLoss(): # pylint: disable=C0103
    """
    Cross entropy loss with particle swarm optimizer cv datasets
    """
    return CrossEntropyLossCreator.CrossEntropyLossWithParticleSwarmOptimizer.apply


def BCELoss(): # pylint: disable=C0103
    """
    Binary cross entropy with particle swarm for tabular datasets
    """
    return (
        CrossEntropyLossCreator.BinaryCrossEntropyLossWithParticleSwarmOptimizer.apply
    )


class CrossEntropyLossCreator:
    """
    Factory class for creating instance of loss functions
    """
    @staticmethod
    def apply_forward(
        ctx, y_targets, y_predictions, swarm_learning_rate, entropy_loss=F.cross_entropy
    ):
        """
        Apply forward propagation to learning weights
        """
        dimension, classes = y_targets.size()
        acceleration_coefficients=AccelerationCoefficients()
        inertial_weight_beta =  0.1
        swarm_size = 10

        particle_swarm_optimizer = ParticleSwarmOptimizer(
            dimension=dimension,
            swarm_size=swarm_size,
            acceleration_coefficients=acceleration_coefficients,
            inertial_weight_beta=inertial_weight_beta,
            number_of_classes=classes,
            targets=y_predictions,
        )
        sum_cr, gbest = particle_swarm_optimizer.run_iteration(number=40)
        ctx.save_for_backward(y_targets, y_predictions)
        ctx.sum_cr = sum_cr
        ctx.eta_learning_rate = swarm_learning_rate
        ctx.gbest = gbest
        return entropy_loss(y_targets, y_predictions)

    @staticmethod
    def apply_backward(ctx, grad_output):
        """
        Apply backward propagation
        """
        y_predicted_label, _ = ctx.saved_tensors
        sum_cr = ctx.sum_cr
        eta_learning_rate = ctx.eta_learning_rate
        grad_input = torch.neg((sum_cr / eta_learning_rate) * (ctx.gbest - y_predicted_label)) # pylint: disable=E1101
        return grad_input, grad_output, None, None, None

    # pylint: disable=W0223
    class CrossEntropyLossWithParticleSwarmOptimizer(torch.autograd.Function):
        """
        Custom loss function with particle swarm optimization
        for CV datasets
        """
        @staticmethod
        def forward(ctx, y, y_pred, swarm_learning_rate=0.1): # pylint: disable=W0221
            return CrossEntropyLossCreator.apply_forward(ctx, y, y_pred, swarm_learning_rate)

        @staticmethod
        def backward(ctx, grad_output): # pylint: disable=W0221
            return CrossEntropyLossCreator.apply_backward(ctx, grad_output)

    class BinaryCrossEntropyLossWithParticleSwarmOptimizer(torch.autograd.Function):
        """
        Custom cross entropy loss function with particle swarm
        optimization for tabular datasets
        """
        @staticmethod
        def forward(ctx, y, y_pred, swarm_learning_rate=0.1): # pylint: disable=W0221
            return CrossEntropyLossCreator.apply_forward(
                ctx, y, y_pred, swarm_learning_rate, F.binary_cross_entropy
            )

        @staticmethod
        def backward(ctx, grad_output): # pylint: disable=W0221
            return CrossEntropyLossCreator.apply_backward(ctx, grad_output)
