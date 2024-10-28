import torch
from typing import Generator, List, Union
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
        """Apply delays to the given parameter state.

        Parameters:
            param (torch.tensor): Current undelayed parameter state
            param_history (torch.tensor): History of past parameter states
            iteration_num (int): Current iteration in the optimization process

        Returns:
            (torch.tensor): Delayed parameter state
            (torch.tensor): Updated parameter history
        """
        pass

class DiscreteDelay(DelayDistribution, metaclass=ABCMeta):
    """Abstract base class for discrete delay distributions.
    
    These delay distributions are defined by the sample method, which returns a 
    tensor of integer delays according to the input size and iteration number.
    """
    @abstractmethod
    def sample(self, size, iteration_num) -> torch.Tensor:
        """Returns a tensor of integer delays according to the input size and
        iteration number.
        """
        pass

    def __call__(self, param, param_history, iteration_num):
        full_param_state = torch.cat([param.detach().unsqueeze(0), param_history], dim=0)
        D = self.sample(param.size(), iteration_num)
        delayed_param = full_param_state.gather(0, D.unsqueeze(0)).squeeze(0)
        return delayed_param, full_param_state[:-1]


class ParallelDiscreteDelay(DelayDistribution, metaclass=ABCMeta):
    """Abstract base class for discrete delay distributions that may not be 
    constant in time, but they always apply the same delay to all parameters 
    at a given iteration.
    """
    @abstractmethod
    def get_delay(self, iteration_num) -> int:
        """Returns the parallel delay length at the given iteration number."""
        pass

    def __call__(self, param, param_history, iteration_num):
        full_param_state = torch.cat([param.detach().unsqueeze(0), param_history], dim=0)
        L = self.get_delay(iteration_num)
        if L < 0:
            raise ValueError(f"Delay length cannot be negative. Got value {L}")
        return full_param_state[L], full_param_state[:-1]

class Uniform(ParallelDiscreteDelay):
    def get_delay(self, iteration_num):
        return self.max_L

class Undelayed(Uniform):
    def __init__(self):
        super().__init__(max_L=0)

class Decaying(ParallelDiscreteDelay):
    def __init__(self, max_L, step_size):
        super().__init__(max_L)

    def get_delay(self, iteration_num):
        return max(0, self.max_L - (iteration_num // self.step_size))

class Stochastic(DiscreteDelay):
    def sample(self, size, iteration_num):
        return torch.randint(0, self.max_L+1, size=size)

