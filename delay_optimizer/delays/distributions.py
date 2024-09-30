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

    def __call__(self, param, param_history, iteration_num):
        raise NotImplementedError("Subclasses must implement the `__call__` method")


class DiscreteDelay(DelayDistribution):
    """Abstract base class for discrete delay distributions.
    
    These delay distributions are defined by the sample method, which returns a tensor
    of integer delays according to the input size and iteration number.

    Calling an instance of a DiscreteDelay object samples a delay distribution for the
    given parameter state and history and applies the delays to the parameter state.
    """
    def sample(self, size, iteration_num):
        raise NotImplementedError("Subclasses must implement the `sample` method")

    def __call__(self, param, param_history, iteration_num):
        """Apply delays to the given parameter state.

        Parameters:
            param (torch.tensor): Current undelayed parameter state
            param_history (torch.tensor): History of past parameter states
            iteration_num (int): Current iteration number in the optimization process

        Returns:
            torch.tensor: Delayed parameter state
            torch.tensor: Updated parameter history
        """
        full_param_state = torch.cat([param.detach().unsqueeze(0), param_history], dim=0)
        if iteration_num >= self.num_delays:    # TODO: Consider using dynamic max_L and param_history
            return param, full_param_state[:-1]
        D = self.sample(param.size(), iteration_num)
        delayed_param = full_param_state.gather(0, D.unsqueeze(0)).squeeze(0)
        return delayed_param, full_param_state[:-1]


class ParallelDiscreteDelay(DiscreteDelay):
    """Abstract base class for discrete delay distributions that may not be constant in 
    time, but they always apply the same delay to all parameters at a given iteration.

    This abstract class reduces the complexity of the `__call__` method for these types
    of delay distributions over the DiscreteDelay class.
    """
    def get_delay(self, iteration_num):
        raise NotImplementedError("Subclasses must implement the `get_delay` method")

    def sample(self, size, iteration_num):
        """This method is not necessary, but is implemented for consistency."""
        L = self.get_delay(iteration_num)
        return torch.full(size, L, dtype=torch.int)

    def __call__(self, param, param_history, iteration_num):
        full_param_state = torch.cat([param.detach().unsqueeze(0), param_history], dim=0)
        L = self.get_delay(iteration_num) if iteration_num < self.num_delays else 0
        if L < 0:
            raise ValueError(f"Delay length cannot be negative. Got value {L}")
        return full_param_state[L], full_param_state[:-1]

        
class Undelayed(ParallelDiscreteDelay):
    def __init__(self):
        super().__init__(max_L=0, num_delays=0)

    def get_delay(self, iteration_num):
        return 0

class Uniform(ParallelDiscreteDelay):
    def get_delay(self, iteration_num):
        if iteration_num < self.num_delays:
            return self.max_L
        return 0

class Decaying(ParallelDiscreteDelay):
    def __init__(self, max_L, num_delays):
        if num_delays == -1:
            raise ValueError("Must specify a positive finite number of delays for decaying delay distributions")
        super().__init__(max_L, num_delays)

    def get_delay(self, iteration_num):
        if iteration_num < self.num_delays:
            return self.max_L - int(iteration_num * self.max_L / self.num_delays)
        return 0

        
class Stochastic(DiscreteDelay):
    def sample(self, size, iteration_num):
        if iteration_num < self.num_delays:
            return torch.randint(0, self.max_L+1, size=size)
        return torch.zeros(size, dtype=int)

class Constant(DiscreteDelay):
    def __init__(self, D: torch.tensor, num_delays: int):
        if torch.is_floating_point(D) or (D < 0).any():
            raise ValueError("Delay distribution D can only contain non-negative integers")
        super().__init__(max_L=torch.max(D).item(), num_delays=num_delays)
        self.D = torch.int(D)

    def sample(self, size, iteration_num):
        if iteration_num < self.num_delays:
            if size != self.D.shape:
                raise ValueError("Constant delay vector D does not match the parameter size")
            else:
                return self.D
        return torch.zeros_like(self.D)
    


