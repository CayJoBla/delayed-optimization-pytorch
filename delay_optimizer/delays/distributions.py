import torch
from abc import ABCMeta, abstractmethod


class DelayDistribution(metaclass=ABCMeta):
    """Abstract base class for delay distributions."""
    def __init__(self, max_L: int):
        self.max_L = max_L

    def __repr__(self):
        return f"{self.__class__.__name__}(max_L={self.max_L})"

    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def __call__(self, param, param_history, iteration_num):
        """Apply delays inplace to the given parameter state.

        Parameters:
            param (torch.tensor): Current undelayed parameter state
            param_history (torch.tensor): History of past parameter states
            iteration_num (int): Current iteration in the optimization process
        """
        pass

class DiscreteDelay(DelayDistribution, metaclass=ABCMeta):
    """Abstract base class for discrete delay distributions.
    
    These delay distributions are defined by the sample method, which returns a 
    tensor of integer delays according to the input size and iteration number.
    """
    @abstractmethod
    def sample(self, size, iteration_num):
        """Returns a tensor of integer delays matching the parameter shape."""
        pass

    def __call__(self, param, param_history, iteration_num):
        param_history.update(param)
        delay_array = self.sample(param.size(), iteration_num)
        param_history.delay_param(delay_array, out=param)


class ParallelDiscreteDelay(DelayDistribution, metaclass=ABCMeta):
    """Abstract base class for discrete delay distributions that may not be 
    constant in time, but they always apply the same delay to all parameters 
    at a given iteration.
    """
    @abstractmethod
    def get_delay(self, iteration_num):
        """Returns the parallel delay length as a 0-dimensional tensor."""
        pass

    def __call__(self, param, param_history, iteration_num):
        param_history.update(param)
        L = self.get_delay(iteration_num)
        if L > 0:
            param.copy_(param_history[L])

        
class Undelayed(DelayDistribution):
    def __init__(self):
        super().__init__(max_L=0)

    def __call__(self, param, param_history, iteration_num):
        if param_history is not None:   # In general, no history for undelayed
            param_history.update(param)
        return param, param_history

class Uniform(ParallelDiscreteDelay):
    def get_delay(self, iteration_num):
        return torch.tensor(self.max_L)

class Stochastic(DiscreteDelay):
    def sample(self, size, iteration_num):
        return torch.randint(0, self.max_L+1, size=size)

class Decaying(ParallelDiscreteDelay):
    def __init__(self, max_L, step_size):
        super().__init__(max_L)
        self.step_size = step_size

    def get_delay(self, iteration_num):
        return torch.tensor(max(0, self.max_L-(iteration_num//self.step_size)))

