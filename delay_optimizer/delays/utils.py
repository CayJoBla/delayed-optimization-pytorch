import torch

class ParamHistoryBuffer:
    """Holds the parameter history and manages delays for a single parameter."""
    def __init__(
        self,
        param,
        buffer_size,
        device = None,
    ):
        self.buffer_size = buffer_size
        self.device = device or param.device
        self._initialize_history(param)

    def __getattr__(self, name):
        return getattr(self._buffer, name)

    def __repr__(self):
        return ("ParamHistoryBuffer(\n"
                f"{self._buffer}, current_idx={self._current_idx})")

    def _initialize_history(self, param):
        """Initialize the history with L copies of the current parameter."""
        self._buffer = param.repeat(self.buffer_size, *(1,)*param.ndim).detach()
        self._current_idx = 0
        
    def _delay_to_idx(self, delay):
        if any(delay <= 0) or any(delay > self.buffer_size):
            raise IndexError(f"Delay must be in [1, {self.buffer_size}]")
        return (self._current_idx + delay - 1) % self.buffer_size

    def __getitem__(self, args):
        delay, *args = args if isinstance(args, tuple) else (args,)
        if isinstance(delay, slice):
             raise ValueError("Slicing in parameter history is not supported "
                                "over the delay dimension.")
        return self._buffer[self._delay_to_idx(delay), *args]

    def update(self, new_param):
        self._current_idx = (self._current_idx - 1) % self.buffer_size
        self._buffer[self._current_idx] = new_param

        