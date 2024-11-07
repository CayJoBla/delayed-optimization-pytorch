import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import inspect
from typing import Union, Type
import warnings

from delay_optimizer.delays.distributions import (
    DelayDistribution,
    Undelayed,
    Uniform,
    Stochastic
)

OPTIMIZER_MAP = {       # TODO: Do I need support for other optimizers?
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD
}

class RunConfig:
    def __init__(
        self,
        optimizer: Union[Type[Optimizer], str] = torch.optim.Adam, 
        delay: Union[Type[DelayDistribution], DelayDistribution, str] = Undelayed, 
        max_L: int = 0,
        batch_size: int = 32,
        num_epochs: int = 10,
        lr_scheduler: Type[LRScheduler] = None,
        do_train: bool = True,
        do_validate: bool = True,
        do_test: bool = False,
        logging_steps: int = 100, 
        do_progress_bar: bool = True, 
        do_wandb_logging: bool = True,
        wandb_project: str = None,
        run_name: str = None, 
        save_dir: str = None,
        **kwargs
    ):
        # Parse optimization parameters
        self.optimizer_class = self._parse_optimizer(optimizer)
        self.optimizer_kwargs = {"delay": self._parse_delay(delay, max_L)}
        self.optimizer_kwargs.update(self._extract_class_kwargs(self.optimizer_class, 
                                                                **kwargs))
        # Parse training parameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr_scheduler = lr_scheduler
        self.do_train = do_train
        self.do_validate = do_validate
        self.do_test = do_test
        self.lr_scheduler_kwargs = self._extract_class_kwargs(self.lr_scheduler, 
                                                                **kwargs)
        # Parse logging parameters
        self.logging_steps = logging_steps
        self.do_progress_bar = do_progress_bar
        self.do_wandb_logging = do_wandb_logging
        self.wandb_project = wandb_project
        self.run_name = run_name
        self.save_dir = save_dir

    @staticmethod
    def _extract_class_kwargs(cls_type, **kwargs):
        if cls_type is None:
            return {}
        param_keys = inspect.signature(cls_type).parameters.keys()
        return {k: v for k, v in kwargs.items() if k in param_keys}

    @staticmethod
    def _parse_optimizer(optimizer):
        if isinstance(optimizer, str):
            optimizer = OPTIMIZER_MAP.get(optimizer.lower(), None)
            if optimizer is None:
                raise ValueError(f"Unrecognized optimizer: {optimizer}")
        return optimizer

    @staticmethod
    def _parse_delay(delay, max_L):
        if isinstance(delay, str):
            if delay == "undelayed":
                if max_L != 0:
                    warnings.warn(f"Specified delay type is 'undelayed', but a "
                                    "nonzero value has been provided for max_L."
                                    " Continuing with undelayed optimization.")
                return Undelayed()
            else:
                if max_L == 0:
                    warnings.warn(f"Specified delay type is '{delay}', but "
                                    "max_L=0. This will result in undelayed "
                                    "optimization.")
                    return Undelayed()
                if delay == "uniform":
                    return Uniform(max_L=max_L)
                elif delay == "stochastic":
                    return Stochastic(max_L=max_L)
                else:
                    raise ValueError(f"Unrecognized delay distribution: {delay}")
        elif issubclass(delay, DelayDistribution):
            # Try parsing via name string, if custom just initialize with max_L
            try:
                return RunConfig._parse_delay(delay.__name__.lower(), max_L)
            except ValueError:
                return delay(max_L=max_L)
        elif isinstance(delay, DelayDistribution):
            return delay

    def to_dict(self):
        optimizer_kwargs = self.optimizer_kwargs.copy()
        delay = optimizer_kwargs.pop("delay")

        return {
            "optimizer": self.optimizer_class.__name__,
            "optimizer_kwargs": optimizer_kwargs,
            "delay": delay.__class__.__name__,
            "max_L": delay.max_L,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "lr_scheduler": self.lr_scheduler.__name__ if self.lr_scheduler else None,
            "lr_scheduler_kwargs": self.lr_scheduler_kwargs,
            "do_train": self.do_train,
            "do_validate": self.do_validate,
            "do_test": self.do_test,
            "logging_steps": self.logging_steps,
            "do_progress_bar": self.do_progress_bar,
            "do_wandb_logging": self.do_wandb_logging,
            "wandb_project": self.wandb_project,
            "run_name": self.run_name,
            "save_dir": self.save_dir
        }
            


