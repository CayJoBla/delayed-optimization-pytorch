# optimization_helper.py 

import numpy as np
import warnings
from tqdm import tqdm

from ..optimization import functions, optimizers, schedulers
from ..delays import DelayedOptimizer, distributions
from .parse import (
    parse_kwargs,
    parse_objective_function,
    parse_scheduler,
    parse_optimizer,
    parse_delay_distribution
)
from .data import Data

class OptimizationHelper:
    """Class for handling delayed or undelayed optimization on a given function"""
    def __init__(self, objective, **kwargs):
        """Initializer for the Handler class
        
        Parameters: 
            objective(str): name of the loss function to be analyzed
            dim(int): the dimension of the state vector
        """
        self.objective = parse_objective_function(objective, **kwargs)

    # Initialize points -----------------------------------------------------

    def random_points(self, num_points):
        """Randomly initialize given number of points within the domain of the objective function"""
        self.x_inits = np.random.uniform(
            *self.objective.domain, 
            size=(num_points, self.objective.n)
        )
        
    def load_points(self, points):
        """Load initial optimization points into the Handler object"""
        points = np.atleast_2d(points)
        if points.shape[1] != self.objective.n:
            raise ValueError(
                "Points array does not match function dimension. Please provide "
                f"an array of points with shape (*,{self.objective.n}).")
        self.x_inits = points
    

    # Run optimization ------------------------------------------------------
   
    def optimize(self, optimizer, delay_type, scheduler="constant", maxiter=5000, 
                 output_dir=None, **kwargs):
        """Run the optimization on the initial points already initialized and 
        saves values to be plotted.
        
        Parameters:
            optimizer(Optimizer,str): the base optimizer
            delay_type(DelayType,str): the delay distribution to apply during optimization
            scheduler(Scheduler,str): the learning rate scheduler to use
            maxiter(int): the maximum number of iterations for optimization
            save_state(bool/tuple): state dimensions to save during optimization
            save_loss(bool): whether to save loss values over time 
            save_grad(bool): whether to save gradient values over time
        """
        # Check if points have been initialized
        if len(self.x_inits) == 0:
            warnings.warn("No points have been initialized.")
            return
        
        # Initialize
        optimizer_kwargs, delay_kwargs, scheduler_kwargs = parse_kwargs(kwargs)
        scheduler = parse_scheduler(scheduler, **scheduler_kwargs)
        optimizer = parse_optimizer(optimizer, lr=scheduler, **optimizer_kwargs)
        delay_type = parse_delay_distribution(delay_type, **delay_kwargs)
        delayer = DelayedOptimizer(self.objective, optimizer, delay_type)

        # Run optimization
        data = Data(self.objective, optimizer, delay_type, maxiter)
        pbar = tqdm(
            range(maxiter),
            desc=r"{} {}d ({})".format(self.objective.__class__.__name__, 
                                        self.objective.n,
                                        delay_type.__class__.__name__),
            leave=True
        )
        delayer.initialize(self.x_inits)
        for i in pbar:
            delayer.step()
            data.update(delayer.time_series[0])

        if output_dir is not None:
            data.save(output_dir)
        else:
            data.condense()
            del data._states
            
        return data
