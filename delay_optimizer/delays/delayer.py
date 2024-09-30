import numpy as np


class DelayedOptimizer:
    """Class that performs delayed optimization on the given objective function 
    with the given optimizer. The sequence of delays to use is determined by the 
    DelayType object.
    """
    def __init__(self, objective, optimizer, delay_type):
        """Initializer for the Delayer class
            
        Parameters:
            objective (ObjectiveFunction): the loss function to be optimized over, 
                including dimension n, loss function, and gradient function
            optimizer (Optimizer): an initialized class object to perform the
                optimization. Must have a step() method that updates the state
                given state and gradient values.
            delay_type (DelayType): object containing delay parameters
        """
        self.objective = objective
        self.optimizer = optimizer
        self.delay_type = delay_type
        
    def initialize(self, x_init):
        x_init = np.atleast_2d(x_init)
        self.optimizer.initialize(x_init)
        self.time_series = np.tile(x_init, (self.delay_type.max_L+1, 1, 1))
        self.delay_generator = self.delay_type.D_gen(x_init.shape)
    
    def step(self):
        """Computes the delayed state and gradient values and takes a step
        with the optimizer.
        """
        # Compute the delayed state and gradient
        D = next(self.delay_generator)              # Get the delay
        del_state = np.take_along_axis(self.time_series, D[np.newaxis,:], axis=0)[0]
        del_grad = self.objective.grad(del_state)
        
        # Roll the time series forward and update
        self.time_series = np.roll(self.time_series, 1, axis=0)
        self.time_series[0] = self.optimizer.step(del_state, del_grad) 

        return self.time_series[0]   
        
    def optimize(self, x_init, maxiter=5000):
        """Computes the time series using the passed optimizer from __init__, 
        saves convergence and time_series which is an array of the states
           
        Parameters:
            x_init (ndarray): the initial state of the calculation
            maxiter (int): the max number of iterations
        Returns:
            (Result): an object containing the optimization data
        """
        self.initialize(x_init) 
        for i in range(maxiter):
            self.step()
        return np.squeeze(self.time_series[0])
