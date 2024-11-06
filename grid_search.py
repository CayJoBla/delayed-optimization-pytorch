import torch
from torch.optim import Adam, SGD
import wandb
import time
import argparse
from itertools import product
import json
import os

from benchmark import tasks
from benchmark.runner import Runner, RunConfig
from delay_optimizer.delays.distributions import Undelayed, Uniform, Stochastic

def run_grid_search(task, optimizer, delay, max_L, lr, momentum, batch_size, 
                    num_epochs, output_dir, do_progress_bar):
    # Get task and training runner
    task = getattr(tasks, task)
    runner = Runner(task)

    # Define non-tunable arguments
    train_config = {
        "optimizer": optimizer,
        "delay": delay,
        "num_epochs": num_epochs,
        "do_train": True,
        "do_validate": True,
        "do_test": False,
        "do_progress_bar": do_progress_bar,
        "do_wandb_logging": True,
        "wandb_project": f"{task.__name__}_grid_search",
    }

    # Define hyperparameter search space
    search_space = {
        "lr": lr,
        "momentum": momentum,
        "batch_size": batch_size,
        "max_L": max_L,
    }

    def grid_search(search_space):
        keys, values = zip(*search_space.items())
        for v in product(*values):
            yield dict(zip(keys, v))
    
    # Run grid search
    results = []
    for hyperparams in grid_search(search_space):
        print(f"Running with hyperparameters: {hyperparams}")
        run_config = RunConfig(**train_config, **hyperparams)
        runner.reset()
        result = runner.run(run_config)
        results.append({
            "params": run_config.to_dict(),
            "results": result,
            "run_dir": runner._run.dir
        })

    # Save results to output_dir
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{task.__name__}_results.json")
        print(f"Saving results to {output_path}")
        with open(output_path, "w") as f:
            json.dump(results, f)

    return results


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Run hyperparameter optimization for a given task."
    )
    argparser.add_argument(
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
        type=str,
        default="undelayed",
        help="The delay distribution to use for training."
    )
    argparser.add_argument(
        "--max_L",
        type=int,
        nargs="*",
        default=[0],
        help="The maximum delay length to use in the delay distribution."
    )
    argparser.add_argument(
        "--lr",
        type=float,
        nargs="*",
        default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        help="The learning rate to use for training."
    )
    argparser.add_argument(
        "--momentum",
        type=float,
        nargs="*",
        default=[0],
        help="The momentum to use for training."
    )
    argparser.add_argument(
        "--batch_size",
        type=int,
        nargs="*",
        default=[64],
        help="The batch size to use for training."
    )
    argparser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="The number of epochs to train for."
    )
    argparser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )
    argparser.add_argument(
        "--disable_progress_bar",
        action="store_false",
        help="Disable progress bar during training.",
        dest="do_progress_bar",
        default=True
    )

    args = argparser.parse_args()
    results = run_grid_search(**vars(args))