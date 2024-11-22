import torch
from torch.optim import Optimizer
from torch.optim.optimizer import _get_scalar_dtype, ParamsT
from typing import Union, Callable, Optional, Type

from .utils import ParamHistoryBuffer
from .distributions import DelayDistribution, Uniform, Undelayed

# TODO: Ideally the application of delays should be done in parallel (with GPU 
#       support), but I would need to look into that more

# TODO: I think the parameter history should probably be saved on RAM not VRAM, 
#       so I should probably check for that


class DelayedOptimizer(Optimizer):
    def __init__(
        self, 
        params: ParamsT,
        optimizer_class: Type[Optimizer],
        *,
        delay: Union[DelayDistribution, int] = 0,
        initial_history: Optional[Callable] = None,
        **optimizer_kwargs
    ):
        if isinstance(delay, int):  # Convert int delays to uniform distribution
            delay = Uniform(max_L=delay) if delay > 0 else Undelayed()
        self._optimizer = optimizer_class(params, **optimizer_kwargs)
        self._optimizer.defaults['delay'] = delay
        self._init_delayed_param_groups()
    
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self._optimizer, name)

    def _init_delayed_param_groups(self):
        """Initialize delay parameters for each parameter group, including 
        delays, parameter histories, and maximal delay length.
        """
        self.max_L = 0      # Max delay lengths over all parameter groups
        default_delay = self._optimizer.defaults["delay"]
        for param_group in self._optimizer.param_groups:
            param_group["delay"] = param_group.get("delay", default_delay)
            L = param_group["delay"].max_L
            param_group["history"] = [
                ParamHistoryBuffer(p, L) for p in param_group["params"]
            ] if L > 0 else None

            if L > self.max_L:
                self.max_L = L

    @torch.no_grad()
    def apply_delays(self, parallelize=False):
        """Applies delays to the parameters being optimized.

        Should be called before the forward pass in order to compute the correct
        gradient and loss values.
        """
        # TODO: Do I need to separate updating the history and applying the delays?
        # TODO: Implement parallelization for applying delays
        for group in self.param_groups:
            if group["delay"].max_L == 0:
                continue
            for i, param in enumerate(group["params"]):
                iteration_num = self.state[param].get("step", torch.tensor(0.))
                param_history = group["history"][i]
                group["delay"](param, param_history, iteration_num) 
                
    def step(self, closure=None):
        return self._optimizer.step(closure)

    