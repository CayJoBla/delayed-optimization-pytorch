import torch
from torch.optim import Adam, SGD, Optimizer
from itertools import product
import yaml
import os
import sys
import numpy as np
import warnings
from deepobs import pytorch as pt

from delay_optimizer.delays.delayed_optimizer import DelayedOptimizer


def get_grid(tunable=False, min_val=None, max_val=None, scale=None, 
                num_samples=None, default=None, dtype=None, **kwargs):
    if tunable:
        if "log" in scale:
            log_base = 10 if scale == "log" else int(scale.split("log")[1])
            min_exp = np.emath.logn(log_base, min_val)
            max_exp = np.emath.logn(log_base, max_val)
            grid = np.logspace(min_exp, max_exp, num_samples, dtype=dtype)
        elif scale == "linear":
            grid = np.linspace(min_val, max_val, num_samples, dtype=dtype)
        else:
            raise ValueError(f"Unknown scale '{scale}'")
    else:
        default = min_val if default is None else default
        default = max_val if default is None else default
        if default is None:
            raise ValueError("Must specify default value for non-tunable hyperparameter.")
        grid = [default]
        
    return grid

def get_param_grid(hyperparam_config):
    hyperparams = {}
    param_grid = {}
    for param, values in hyperparam_config.items():
        print(param)
        print(values)
        if param == "delay":
            if values.get("dtype", "dict") != "dict":
                raise warnings.warn(f"Delay hyperparameter should be of type 'dict', not '{values['dtype']}'.")
            if values.get("tunable", False):
                raise NotImplementedError("Tunable delay hyperparameters are not yet supported.")
            hyperparams[param] = {"type": dict}
            param_grid[param] = [{
                "delay_type": values["delay_type"], 
                "max_L": values["max_L"]
            }]
            continue
    
        hyperparams[param] = {"type": values.get("dtype", float)}
        param_grid[param] = get_grid(**values)

    return hyperparams, param_grid


def run_grid_search(config_file=None):
    # Load the config
    if config_file is None:
        raise ValueError("Must specify job configuration file for hyperparameter optimization.")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Get delayed optimizer class
    if config["optimizer"] == "adam":
        optimizer = Adam
    elif optimizer == "sgd":
        optimizer = SGD
    else:
        raise ValueError(f"'optimizer' parameter not recognized: {config['optimizer']}.")
    delayed_opt_class = DelayedOptimizer(optimizer)

    # Get hyperparameter grid
    hyperparams, param_grid = get_param_grid(config["hyperparams"])
    if config["optimizer"] == "adam":
        hyperparams.pop("momentum")
        param_grid.pop("momentum")
    delay = param_grid.pop("delay")[0]

    # Apply product over grid search space
    def grid_search(search_space):
        keys, values = zip(*search_space.items())
        for v in product(*values):
            yield dict(zip(keys, v))

    # Initialize runner
    runner = pt.runners.StandardRunner(delayed_opt_class, hyperparams)
    
    # Run grid search
    for params in grid_search(param_grid):
        params["delay"] = {"delay_type": delay["delay_type"], "max_L": delay["max_L"]}
        print(f"Running with hyperparameters: \n{params}\n")
        runner.run(
            testproblem=config["testproblem"], 
            hyperparams=params, 
            num_epochs=config.get("num_epochs", 1),
            batch_size=config.get("batch_size", 32),
        )


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     raise ValueError(f"Usage: python {sys.argv[0]} <job_config_filepath>")
    # results = run_grid_search(sys.argv[1])
    CONFIG = "config.yaml"
    run_grid_search(CONFIG)
