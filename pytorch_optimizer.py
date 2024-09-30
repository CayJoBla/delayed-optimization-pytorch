import torch
from torch.optim import Optimizer
from torch.optim.optimizer import _get_scalar_dtype, ParamsT

from typing import Union, Callable, Optional, Type


# TODO: Ideally the application of delays should be done in parallel (with GPU 
#       support), but I would need to look into that more

# TODO: I think the parameter history should probably be saved on RAM not VRAM, 
#       so I should probably check for that


def get_constant_delay_func(L):
    def constant_delay_update(param, param_history):
        full_param_state = torch.cat([param.clone().detach().unsqueeze(0), 
                                        param_history], dim=0)
        return full_param_state[L], full_param_state[:-1]
    return constant_delay_update


class DelayedOptimizer(Optimizer):
    def __init__(
        self, 
        params: ParamsT,
        optimizer_class: Type[Optimizer],
        *,
        delay: Union[int, Callable] = 0,
        max_L: Optional[int] = None,
        initial_history: Optional[torch.Tensor] = None,
        **optimizer_kwargs
    ):
        self._optimizer = optimizer_class(params, **optimizer_kwargs)
        self._optimizer.defaults['delay'] = delay
        self._optimizer.defaults['max_L'] = max_L
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
        L = param_group["max_L"]
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
        max_L = 0
        for param_group in self._optimizer.param_groups:
            delay = param_group.get("delay", self._optimizer.defaults["delay"])
            params = param_group["params"]

            if isinstance(delay, int):
                L = delay
                if L < 0:
                    raise ValueError("Delay length must be non-negative")
                delay = get_constant_delay_func(L)
            elif callable(delay):
                if hasattr(delay, 'max_L'):
                    L = delay.max_L
                elif "max_L" in param_group:
                    L = param_group["max_L"]
                elif self._optimizer.defaults["max_L"] is not None:
                    L = self._optimizer.defaults["max_L"]
                else:
                    raise TypeError("Delay length parameter 'max_L' not "
                                    "specified for callable delay")
            else:
                raise TypeError(f"Invalid delay type: {type(delay)}")

            param_group["max_L"] = L
            param_group["delay"] = delay
            initial_history = param_group.get("init_history", 
                                self._optimizer.defaults["init_history"])
            initial_history(param_group)

            # Check the size of the delay history
            for i in range(len(params)):
                param = params[i]
                param_history = param_group["history"][i]
                if param_history.shape != (L,)+param.shape:
                    raise ValueError("Invalid parameter history shape: "
                                    f"{tuple(param_history.shape)} where size "
                                    f"{(L,)+param.shape} was expected.")
            if L > max_L:
                max_L = L

        self.max_L = max_L

    def apply_delays(self):
        """My current vision is that the user would call this function before 
        running a forward pass through the model, similar to `zero_grad`. This 
        function applies delays to the parameters in the model as specified 
        during initialization. 

        Alternatively, delays could be applied in the 'zero_grad' function, but 
        I think that falls away from the purpose of that function. Furthermore, 
        the separation of this method allows the user to follow a custom delay 
        schedule, if desired.

        This function should be called before the forward pass through the model 
        in order to compute the correct gradient and loss values.
        """
        # TODO: Implement parallelization for applying delays
        for group in self.param_groups:
            if group["max_L"] == 0:     # No delays, no history to update
                continue
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

    