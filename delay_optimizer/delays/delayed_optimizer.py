import torch
from torch.optim import Optimizer
from torch.optim.optimizer import _get_scalar_dtype, ParamsT
from typing import Union, Callable, Optional, Type

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
        if initial_history is None:
            self._optimizer.defaults['init_history'] = self._init_param_history
        else:
            self._optimizer.defaults['init_history'] = initial_history
        
        self._init_delayed_param_groups()
    
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self._optimizer, name)

    def _init_param_history(self, param_group):
        """Default parameter history initialization. 

        Default behavior is to initialize the history with L copies of the 
        current parameter value, or an empty tensor if L=0.
        """
        L = param_group["delay"].max_L
        if L == 0:
            history = [torch.empty(0, *p.size()) for p in param_group["params"]]
        else:
            history = [torch.stack([p.clone().detach() for _ in range(L)],
                                    dim=0) for p in param_group["params"]]
        param_group["history"] = history

    def _init_delayed_param_groups(self):
        """Initialize delay parameters for each parameter group, including past 
        parameters and maximal delay length, for each parameter group.
        """
        self.max_L = 0
        for param_group in self._optimizer.param_groups:
            param_group["delay"] = param_group.get("delay", 
                                    self._optimizer.defaults["delay"])
            initial_history = param_group.get("init_history", 
                                self._optimizer.defaults["init_history"])
            initial_history(param_group)    # TODO: Is this the baest way to do this?

            # Check the size of the delay history
            params = param_group["params"]
            for i in range(len(params)):
                param = params[i]
                param_history = param_group["history"][i]
                if param_history.shape != (L,)+param.shape:
                    raise ValueError("Invalid parameter history shape: "
                                    f"{tuple(param_history.shape)} where size "
                                    f"{(L,)+param.shape} was expected.")

            # Get the maximal delay length over all parameter groups
            L = param_group["delay"].max_L
            if L > self.max_L:
                self.max_L = L

    def apply_delays(self):
        """Applies delays to the parameters being optimized.

        Should be called before the forward pass in order to compute the correct
        gradient and loss values.
        """
        # TODO: Implement parallelization for applying delays
        for group in self.param_groups:
            for i, (param, param_history) in enumerate(zip(group["params"],
                                                            group["history"])):
                iteration_num = self.state[param].get(
                    "step", 
                    torch.tensor(0.0, dtype=_get_scalar_dtype())
                )
                with torch.no_grad(): 
                    delayed_param, updated_history = group["delay"](
                        param, 
                        param_history, 
                        iteration_num
                    )
                    param.copy_(delayed_param)
                    param_history.copy_(updated_history)

    def step(self, closure=None):
        return self._optimizer.step(closure)

    