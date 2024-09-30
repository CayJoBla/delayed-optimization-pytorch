import numpy as np
import math

class Scheduler:
    def __init__(self):
        self._t = 0

    def __iter__(self):
        return self
        
    def __next__(self):
        self._t += 1
        return self.schedule(self._t)

    def __getitem__(self, t):
        return self.schedule(t)

    def current(self):
        return self.schedule(self._t)

    def schedule(self, t):
        raise NotImplementedError("Subclasses must implement this method")


class Constant(Scheduler):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr

    def schedule(self, t):
        return self.lr

    def get_params(self):
        return {"lr": self.lr}
        

class Step(Scheduler):
    def __init__(self, max_lr, gamma, step_size):
        super().__init__()
        self.max_lr = max_lr
        self.gamma = gamma
        self.step_size = step_size
        
    def schedule(self, t):
        return self.max_lr * (self.gamma ** (t // self.step_size))

    def get_params(self):
        return {"max_lr": self.max_lr, "gamma": self.gamma, "step_size": self.step_size}
    

class Inv(Scheduler):
    def __init__(self, max_lr, gamma, p):
        super().__init__()
        self.max_lr = max_lr
        self.gamma = gamma
        self.p = p

    def schedule(self, t):
        return self.max_lr * ((1 + t * self.gamma) ** -self.p)

    def get_params(self):
        return {"max_lr": self.max_lr, "gamma": self.gamma, "p": self.p}
     

class Tri2(Scheduler):
    def __init__(self, max_lr, min_lr, step_size):
        super().__init__()
        self.max_lr = max(max_lr, min_lr)
        self.min_lr = min(max_lr, min_lr)
        self.step_size = step_size
        self._width = self.max_lr - self.min_lr

    def schedule(self, t):
        val1 = t / (2 * self.step_size)
        val2 = 2 / math.pi * abs(math.asin(math.sin(math.pi * val1)))
        return self.min_lr + val2 * (self._width / 2**math.floor(val1))

    def get_params(self):
        return {"max_lr": self.max_lr, "min_lr": self.min_lr, "step_size": self.step_size}


class Sin2(Scheduler):
    def __init__(self, max_lr, min_lr, step_size):
        super().__init__()
        self.max_lr = max(max_lr, min_lr)
        self.min_lr = min(max_lr, min_lr)
        self.step_size = step_size
        self._width = self.max_lr - self.min_lr

    def schedule(self, t):
        val1 = t / (2 * self.step_size)
        val2 = abs(math.sin(math.pi * val1))
        return self.min_lr + val2 * (self._width / 2**math.floor(val1))

    def get_params(self):
        return {"max_lr": self.max_lr, "min_lr": self.min_lr, "step_size": self.step_size}
