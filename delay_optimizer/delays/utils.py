import torch

class ParamHistoryBuffer:
    """Holds the parameter history and indexes delays for a single parameter."""
    def __init__(self, param, max_L):
        self.buffer_size = max_L + 1    # History (L,) + param (1,)
        self._buffer = param.detach().repeat(self.buffer_size, *(1,)*param.ndim)
        self._current_idx = 0

    def __getattr__(self, name):
        return getattr(self._buffer, name)

    def __repr__(self):
        return ("ParamHistoryBuffer(\n"
                f"{self._buffer}, current_idx={self._current_idx})")

    def _delay_to_idx(self, delay):
        max_L = self.buffer_size - 1
        delay = torch.as_tensor(delay)
        if torch.any(delay < 0) or torch.any(delay > max_L):
            raise IndexError(f"Delay must be in [0, {max_L}]")
        return (self._current_idx + delay) % self.buffer_size

    def __getitem__(self, args):
        delay, *args = args if isinstance(args, tuple) else (args,)
        if isinstance(delay, slice):
            raise ValueError("Slicing not supported over the delay dimension.")
        return self._buffer[self._delay_to_idx(delay), *args]

    def update(self, new_param):
        """Adds the most recent parameter value to the history buffer."""
        self._current_idx = (self._current_idx - 1) % self.buffer_size
        self._buffer[self._current_idx] = new_param

    def delay_param(self, delay, out=None):
        """Gather the delayed parameter from the history buffer for the given
        delay matrix.
        """
        return self._buffer.gather(0, self._delay_to_idx(delay), out=out)
        