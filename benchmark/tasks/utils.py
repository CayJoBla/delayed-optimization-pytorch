import torch
from torch.optim import lr_scheduler
import inspect

class OptimizerConfig:
    def __init__(
        self, 
        optimizer = torch.optim.Adam, 
        delay = 0, 
        max_L = None, 
        **kwargs
    ):
        self.optimizer_class = optimizer
        self.optimizer_kwargs = {
            "delay": delay,
            "max_L": max_L,
        }

        # Get other optimizer hyperparameters
        hyperparam_keys = inspect.signature(optimizer).parameters.keys()
        optimizer_params = {k: v for k, v in kwargs.items() if k in hyperparam_keys}
        self.optimizer_kwargs.update(optimizer_params)

class TrainConfig:
    def __init__(
        self,
        batch_size = 32,
        num_epochs = 10,
        lr_scheduler = None,
        do_train = True,
        do_validate = True,
        **kwargs
    ):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr_scheduler = lr_scheduler
        self.do_train = do_train
        self.do_validate = do_validate

        # Get other lr_scheduler hyperparameters
        self.lr_scheduler_kwargs = {}
        if lr_scheduler is not None:
            param_keys = inspect.signature(lr_scheduler).parameters.keys()
            optimizer_params = {k: v for k, v in kwargs.items() if k in param_keys}
            self.lr_scheduler_kwargs.update(optimizer_params)

class LoggingConfig:
    def __init__(
        self, 
        logging_steps = 100, 
        do_progress_bar = True, 
        wandb_project = None, 
        run_name = None, 
        save_dir = None,
        **kwargs
    ):
        self.logging_steps = logging_steps
        self.do_progress_bar = do_progress_bar
        self.wandb_project = wandb_project
        self.run_name = run_name
        self.save_dir = save_dir

def parse_configs(**kwargs):
    return OptimizerConfig(**kwargs), TrainConfig(**kwargs), LoggingConfig(**kwargs)
