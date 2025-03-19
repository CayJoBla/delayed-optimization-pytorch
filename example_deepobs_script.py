"""Example run script for PyTorch and a delayed optimizer"""

from torch.optim import Optimizer, Adam
from deepobs import pytorch as pt
from typing import Type, Union

from delay_optimizer.delays.delayed_optimizer import DelayedOptimizer
from delay_optimizer.delays.distributions import DelayDistribution, Stochastic

optimizer_class = DelayedOptimizer(Adam)
hyperparams = {
    "lr": {"type": float},
    "delay": {"type": Union[dict]}
}

runner = pt.runners.StandardRunner(optimizer_class, hyperparams)
runner.run(
    testproblem='mnist_2c2d', 
    hyperparams={
        'lr': 1e-3,
        'delay': {
            'delay_type': 'stochastic',
            'max_L': 1
        }
    }, 
    num_epochs=1,
    batch_size=64,
)

runner.run(
    testproblem='mnist_2c2d', 
    hyperparams={
        'lr': 1e-4,
        'delay': {
            'delay_type': 'undelayed',
            'max_L': 0
        }
    }, 
    num_epochs=1,
    batch_size=32,
)