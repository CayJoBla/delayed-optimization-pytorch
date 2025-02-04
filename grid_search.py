import torch
from torch.optim import Adam, SGD, Optimizer
import wandb
import time
import argparse
from itertools import product
import yaml
import os
from typing import Type, Union
import numpy as np
import warnings

from deepobs import pytorch as pt

from delay_optimizer.delays.delayed_optimizer import DelayedOptimizer
from delay_optimizer.delays.distributions import Undelayed, Stochastic


def get_param_grid(config):
    dtype = config['type']
    min_val = config['min']
    max_val = config['max']
    scale = config['scale']
    num_samples = config['num_samples']

    if "log" in scale:
        log_base = 10 if scale == "log" else int(scale.split("log")[1])
        min_exp = np.emath.logn(log_base, min_val)
        max_exp = np.emath.logn(log_base, max_val)
        param_grid = np.logspace(min_exp, max_exp, num_samples, dtype=dtype)
    elif scale == "linear":
        param_grid = np.linspace(min_val, max_val, num_samples, dtype=dtype)
    else:
        raise ValueError(f"Unknown scale '{scale}'")
    return param_grid


def run_grid_search(testproblem, optimizer, delay, max_L, lr, momentum, 
                    config_file, tunable_params, batch_size, num_epochs):
    # Get delayed optimizer class
    if optimizer == "adam":
        optimizer = Adam
    elif optimizer == "sgd":
        optimizer = SGD
    elif not issubclass(optimizer, torch.optim.Optimizer):
        raise ValueError("'optimizer' must be 'adam', 'sgd', or a torch.optim.Optimizer subclass.")
    delayed_opt_class = DelayedOptimizer(optimizer)

    # Define tunable hyperparameters
    tune_lr = False
    tune_momentum = False
    for param in tunable_params:
        if param == "lr":
            tune_lr = True
        elif param == "momentum":
            tune_momentum = True
        else:
            raise ValueError(f"Unrecognized tunable hyperparameter: '{param}'")

    # Check for errors and determine whether to load the config
    if lr is not None:
        if not tune_lr and len(lr) > 1:
            warnings.warn("'lr' not specified as a tunable parameter, ignoring additional values.")
            lr = lr[:1]
    elif not tune_lr:
        raise ValueError("Must specify 'lr' as a tunable parameter and/or provide value(s) for 'lr'.")
    if momentum is not None:
        if not tune_momentum and len(momentum) > 1:
            warnings.warn("'momentum' not specified as a tunable parameter, ignoring additional values.")
            momentum = momentum[:1]
    
    lr_from_config = (tune_lr and lr is None)
    momentum_from_config = (tune_momentum and momentum is None)
    load_config = lr_from_config or momentum_from_config
    if not load_config and config_file is not None:
        warning.warn("Ignoring 'config_file' since all tunable hyperparameters are specified.")

    # Initialize parameter grid
    param_grid = {}
    if lr is not None:
        param_grid["lr"] = np.array(lr)
    if momentum is not None:
        param_grid["momentum"] = np.array(momentum)

    # Load hyperparameter grid from config
    if load_config:
        if config_file is None:
            raise ValueError("Must specify 'config_file' if not all tunable hyperparameters are specified.")
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)['hyperparams']
        if lr_from_config:
            if "lr" not in config:
                raise ValueError("Tunable parameter 'lr' is not specified either in the config file or as an argument.")
            param_grid["lr"] = get_param_grid(config["lr"])
            
        if momentum_from_config:
            if "momentum" not in config:
                raise ValueError("Tunable parameter 'momentum' is not specified either in the config file or as an argument.")
            param_grid["momentum"] = get_param_grid(config["momentum"])

    # Check parameter grid and define hyperparams
    if "lr" not in param_grid:
        raise ValueError("No learning rates specified in the parameter grid.")
    hyperparams = {param: {"type": float} for param in param_grid.keys()}

    # Add delay hyperparameters (not tunable)
    hyperparams["delay"] = {"type": dict}
    param_grid["delay"] = [{"delay_type": delay, "max_L": max_L}]

    # Apply product over grid search space
    def grid_search(search_space):
        keys, values = zip(*search_space.items())
        for v in product(*values):
            yield dict(zip(keys, v))

    # Initialize runner
    runner = pt.runners.StandardRunner(delayed_opt_class, hyperparams)
    
    # Run grid search
    for params in grid_search(param_grid):
        print(f"Running with hyperparameters: \n{params}\n")
        # runner.run(
        #     testproblem=testproblem, 
        #     hyperparams=params, 
        #     num_epochs=num_epochs,
        #     batch_size=batch_size,
        # )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Run hyperparameter optimization for a given task."
    )
    argparser.add_argument(
        "--testproblem",
        "--task",
        type=str,
        default=None,
        help="The name of the task to train on for hyperparameter optimization."
    )
    argparser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="The optimizer to use for training."
    )
    argparser.add_argument(
        "--delay",
        "--delay_type",
        type=str,
        default="undelayed",
        help="The delay distribution to use for training."
    )
    argparser.add_argument(
        "--max_L",
        type=int,
        default=0,
        help="The maximum delay length to use in the delay distribution."
    )
    argparser.add_argument(
        "--lr",
        "--learning_rate",
        type=float,
        nargs="*",
        default=None,
        help="The learning rate or list of learning rates to use for optimization / grid search."
    )
    argparser.add_argument(
        "--momentum",
        type=float,
        nargs="*",
        default=None,
        help="The momentum or list of momentum values to use for optimization / grid search."
    )
    argparser.add_argument(
        "--config_file",
        "--config",
        type=str,
        default=None,
        help="The path to the configuration file for hyperparameter optimization."
    )
    argparser.add_argument(
        "--tunable_params",
        "--tunable",
        type=str,
        nargs="*",
        default=["lr"],
        help="The hyperparameters to tune in the grid search."
    )
    argparser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="The batch size to use for training."
    )
    argparser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="The number of epochs to train for."
    )

    args = argparser.parse_args()
    results = run_grid_search(**vars(args))