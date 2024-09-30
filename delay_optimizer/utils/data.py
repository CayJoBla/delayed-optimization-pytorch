# Data.py

import numpy as np
import pickle
import blosc

from .parse import (
    parse_objective_function,
    parse_optimizer,
    parse_scheduler,
    parse_delay_distribution
)


class Data:
    """Object to hold optimization data."""
    
    def __init__(self, objective, optimizer, delay_type, maxiter):
        # Set values from input objects
        self.objective = objective

        self.optimizer = optimizer.__class__.__name__.lower()
        self.optimizer_params = {k:v for k,v in optimizer.__dict__.items() if k not in {'lr','initialized'}}

        self.scheduler = optimizer.lr.__class__.__name__.lower()
        self.scheduler_params = optimizer.lr.get_params()

        self.delay_type = delay_type.__class__.__name__.lower()
        self.delay_params = delay_type.__dict__

        self.maxiter = maxiter
        self._states = []    # Running state values (compressed to 2d when saved)
        self.state_vals = None
        self.loss_vals = None
    

    # Optimization ---------------------------------------------------------

    def update(self, X):
        self._states.append(X)
        if (len(self._states)+1) * X.size > 1e8:
            self.condense()

    # Data retrieval -------------------------------------------------------
    
    def get_objective(self):
        return parse_objective_function(self.objective, **self.objective_params)

    def get_optimizer(self):
        lr = self.get_scheduler()
        return parse_optimizer(self.optimizer, lr=lr, **self.optimizer_params)
    
    def get_delay_type(self):
        return parse_delay_distribution(self.delay_type, **self.delay_params) 

    def get_scheduler(self):
        return parse_scheduler(self.scheduler, **self.scheduler_params)

    
    # Saving / Loading -----------------------------------------------------
    
    def condense(self):
        """Computes the loss values and purges the running state values, saving only 
        the first two dimensions for plotting data.

        This method is called when the number of state values is too large or when 
        data is being saved to a file.
        """
        if len(self._states) == 0:   # No data to condense
            return

        X = np.array(self._states)
        state_vals = X[...,:2]
        loss_vals = self.objective.loss(X.reshape(-1, X.shape[-1])).reshape(*X.shape[:-1])

        if self.state_vals is None:
            self.state_vals = state_vals.astype(np.float32)
            self.loss_vals = loss_vals.astype(np.float32)
        else:
            self.state_vals = np.concatenate((self.state_vals, state_vals), axis=0)
            self.loss_vals = np.concatenate((self.loss_vals, loss_vals), axis=0)

        self._states = []
        
    def save(self, filename): 
        """Save data to given file"""
        if "_states" in self.__dict__:
            self.condense()         # Condense before saving
            del self._states

        # Parse objective function
        if not isinstance(self.objective, str):
            self.objective_params = {k:v for k,v in self.objective.__dict__.items() if k != 'minimizer'}
            self.objective = self.objective.__class__.__name__.lower()
        
        if not filename.endswith('.dat'):   # Format filename
            filename += '.dat'
            
        pickled_data = pickle.dumps(self)
        compressed_pickle = blosc.compress(pickled_data)
        
        with open(filename, "wb") as file: 
            file.write(compressed_pickle)
        
    @classmethod
    def load(cls, filename):
        """Load data from file and return class object"""
        if not filename.endswith('.dat'):   # Format filename
            filename += '.dat'
        
        with open(filename, "rb") as f:
            compressed_pickle = f.read()

        pickled_data = blosc.decompress(compressed_pickle)
        data = pickle.loads(pickled_data)
        
        if type(data) is not cls:
            raise ValueError("Invalid data file.")
        return data
    
    