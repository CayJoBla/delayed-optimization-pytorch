import numpy as np
from .schedulers import Constant

class Optimizer:
    def __init__(self, lr=0.01):
        if isinstance(lr, (float, int)):
            lr = Constant(lr)
        self.lr = lr
        self.initialized = False

    def initialize(self, x_init):
        raise NotImplementedError("Subclasses must implement this method")

    def step(self, x, grad):
        raise NotImplementedError("Subclasses must implement this method")

    def __call__(self, x, grad):
        return self.step(x, grad)
       

class GradientDescent(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__(lr)

    def initialize(self, x_init):
        self.initialized = True

    def step(self, x, grad):
        return x - next(self.lr) * grad


class Momentum(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9,):
        super().__init__(lr)
        self.momentum = momentum
        
    def initialize(self, x_init):
        self.v = np.zeros_like(x_init)
        self.initialized = True
    
    def step(self, x, grad):
        self.v = self.momentum * self.v + next(self.lr) * grad
        return x - self.v

      
class Adam(Optimizer):
    def __init__(self, lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        super().__init__(lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def initialize(self, x_init):
        self.m = np.zeros_like(x_init)
        self.v = np.zeros_like(x_init)
        self._t = 0
        self.initialized = True
               
    def step(self, x, grad):
        self._t += 1
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.square(grad)
        m_hat = self.m / (1 - self.beta_1 ** self._t)
        v_hat = self.v / (1 - self.beta_2 ** self._t)
        return x - (next(self.lr) *  m_hat) / (np.sqrt(v_hat) + self.epsilon)
