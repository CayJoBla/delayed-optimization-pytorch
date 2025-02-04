import torch
from torch.optim import Optimizer
from torch.optim.optimizer import _get_scalar_dtype, ParamsT
from typing import Union, Callable, Optional, Type

from .distributions import DelayDistribution, Uniform, Undelayed

# TODO: Ideally the application of delays should be done in parallel (with GPU 
#       support), but I would need to look into that more

# TODO: I think the parameter history should probably be saved on RAM not VRAM, 
#       so I should probably check for that

def DelayedOptimizer(base_optimizer_class: Type[Optimizer]):
    """Returns a new optimizer class that wraps a given optimizer class to
    implement delayed optimization on that optimization algorithm.
    """

    class DelayedOptimizerWrapper(base_optimizer_class):
        """Implements delayed optimization for a given optimizer class."""
        base_optimizer = base_optimizer_class

        def __init__(
            self, 
            params: ParamsT,
            *args,
            delay: Union[DelayDistribution, dict, int] = 0,
            init_history: Optional[Callable] = None,
            **kwargs
        ):
            super().__init__(params, *args, **kwargs)

            # Initialize default delay distribution
            self.defaults['delay'] = self._parse_delay(delay)

            # Initialize default parameter history initialization function
            if init_history is None:
                init_history = self._init_param_history
            self.defaults['init_history'] = init_history
            
            self._init_delayed_param_groups()   # Initialize parameter histories

        def __repr__(self):
            return f"Delayed{super().__repr__()}"

        def __str__(self):
            return f"Delayed{super().__str__()}"

        @staticmethod
        def _parse_delay(delay):
            if isinstance(delay, DelayDistribution):
                return delay
            elif isinstance(delay, int):
                return Uniform(max_L=delay) if delay > 0 else Undelayed()
            elif isinstance(delay, dict):  # Convert int delays to uniform distribution
                if "delay_type" not in delay:
                    raise ValueError("Must specify delay type in delay parameter dictionary.")
                if "max_L" not in delay:
                    if delay["delay_type"] != "undelayed":
                        raise ValueError("Must specify max_L in delay parameter dictionary.")
                return DelayDistribution.from_dict(delay)
            else:
                raise ValueError(f"Invalid delay parameter type: {type(delay)}.")

        @staticmethod
        def _init_param_history(param_group):
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
            for param_group in self.param_groups:
                param_group["delay"] = param_group.get("delay", 
                                                        self.defaults["delay"])
                L = param_group["delay"].max_L
                init_history = param_group.get("init_history", 
                                                self.defaults["init_history"])
                init_history(param_group)    # TODO: Is this the best way to do this?

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

    DelayedOptimizerWrapper.__name__ = f"Delayed{base_optimizer_class.__name__}"

    return DelayedOptimizerWrapper

        