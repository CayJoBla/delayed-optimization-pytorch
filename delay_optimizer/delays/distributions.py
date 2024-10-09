import torch
from typing import Generator, List, Union

class DelayDistribution():
    """Abstract base class for delay distributions."""
    def __init__(self, max_L: int, num_delays: int = -1):
        self.max_L = max_L
        self.num_delays = num_delays if num_delays != -1 else float("inf")

    def __repr__(self):
        return f"{self.__class__.__name__}(max_L={self.max_L}, num_delays={self.num_delays})"

    def __str__(self):
        return self.__class__.__name__

    def is_undelayed(self, iteration_num):
        return (iteration_num >= self.num_delays) or (self.max_L == 0)

    @staticmethod
    def undelayed_update(param, param_history):
        if param_history is not None:
            param_history.update(param)
        return param, param_history

    @staticmethod
    def undelayed_check(function):
        def wrapper(self, param, param_history, iteration_num):
            if self.is_undelayed(iteration_num):
                return DelayDistribution.undelayed_update(param, param_history)
            return function(self, param, param_history, iteration_num)
        return wrapper

class DiscreteDelay(DelayDistribution):
    """Abstract base class for discrete delay distributions.
    
    These delay distributions are defined by the sample method, which returns a 
    tensor of integer delays according to the input size and iteration number.

    Calling an instance of a DiscreteDelay object samples a delay distribution 
    for the given parameter state and history and applies the delays to the 
    parameter state.
    """
    def sample(self, size, iteration_num):
        raise NotImplementedError("Subclasses must implement the `sample` method")

    @DelayDistribution.undelayed_check
    def __call__(self, param, param_history, iteration_num):
        """Apply delays to the given parameter state.

        Parameters:
            param (torch.tensor): Current undelayed parameter state
            param_history (torch.tensor): History of past parameter states
            iteration_num (int): Current iteration in the optimization process

        Returns:
            (torch.tensor): Delayed parameter state
            (torch.tensor): Updated parameter history
        """
        delay_array = self.sample(param.size(), iteration_num)
        delayed_mask = (delay_array > 0)
        if not delayed_mask.any():
            return DelayDistribution.undelayed_update(param, param_history)
        delayed_values = param_history[delay_array[delayed_mask], 
                            *delayed_mask.nonzero(as_tuple=True)]
        param_history.update(param)
        param[delayed_mask] = delayed_values
        return param, param_history

class ParallelDiscreteDelay(DelayDistribution):
    """Abstract base class for discrete delay distributions that may not be constant in 
    time, but they always apply the same delay to all parameters at a given iteration.

    This abstract class reduces the complexity of the `__call__` method for these types
    of delay distributions over the DiscreteDelay class.
    """
    def get_delay(self, iteration_num):
        raise NotImplementedError("Subclasses must implement the `get_delay` method")

    @DelayDistribution.undelayed_check
    def __call__(self, param, param_history, iteration_num):
        L = self.get_delay(iteration_num)
        if L == 0:
            return DelayDistribution.undelayed_update(param, param_history)
        delayed_values = param_history[L]
        param_history.update(param)
        param.copy_(delayed_values)
        return param, param_history
        
class Undelayed(DelayDistribution):
    """Distribution that applies no delays to the parameter state."""
    def __init__(self):
        super().__init__(max_L=0, num_delays=0)

    def __call__(self, param, param_history, iteration_num):
        # NOTE: This assumes that the parameter history is always empty
        return param, param_history

class Uniform(ParallelDiscreteDelay):
    def get_delay(self, iteration_num):
        return self.max_L
        
class Stochastic(DiscreteDelay):
    def sample(self, size, iteration_num):
        return torch.randint(0, self.max_L+1, size=size)

class Constant(DiscreteDelay):
    def __init__(self, D: torch.tensor, num_delays: int):
        if torch.is_floating_point(D) or (D < 0).any():
            raise ValueError("Delay distribution D can only contain non-negative integers")
        super().__init__(max_L=torch.max(D).item(), num_delays=num_delays)
        self.D = torch.int(D)

    def sample(self, size, iteration_num):
        return self.D
    


