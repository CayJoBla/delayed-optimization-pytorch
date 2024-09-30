import numpy as np
from typing import Tuple, Callable
import numpy as np

class ObjectiveFunction:
    def __init__(self, n: int, domain: Tuple[float], minimizer: np.ndarray):
        self.n = n
        self.domain = domain
        self.minimizer = minimizer

    def get_loss_function(self) -> Callable:
        raise NotImplementedError("Subclasses must implement this method")

    def get_grad_function(self) -> Callable:
        raise NotImplementedError("Subclasses must implement this method")
    
    def loss(self, x) -> float:
        return self.get_loss_function()(x)

    def grad(self, x) -> np.ndarray[float]:
        return self.get_grad_function()(x)
    
    def __call__(self, x) -> float:
        return self.loss(x)

    def __repr__(self):
        return f"{self.__class__.__name__}(n={self.n}, domain={self.domain})"

    def __str__(self):
        return self.__class__.__name__


class Ackley(ObjectiveFunction):
    def __init__(
        self, 
        n: int, 
        domain: Tuple[float] = (-32.768, 32.768), 
        a: float = 20., 
        b: float = 0.2, 
        c: float = 2*np.pi
    ):
        super().__init__(n, domain=domain, minimizer=np.zeros(n))
        self.a = a
        self.b = b
        self.c = c

    def get_loss_function(self) -> Callable:
        def ackley(x: np.ndarray) -> float:
            sum_x2 = np.sum(np.square(x), axis=-1)
            part_1 = np.exp(-self.b * np.sqrt(sum_x2 / self.n))
            part_2 = np.exp(np.sum(np.cos(self.c * x), axis=-1) / self.n)
            return -self.a * part_1 - part_2 + self.a + np.e
        return ackley

    def get_grad_function(self) -> Callable:
        def ackley_grad(x: np.ndarray) -> np.ndarray[float]:
            sum_x2 = np.sum(np.square(x), axis=-1)
            part_1a = self.a * self.b * x.T / np.sqrt(self.n * sum_x2)
            part_1b = np.exp(-self.b * np.sqrt(sum_x2 / self.n))
            part_2a = self.c * np.sin(self.c * x) / self.n
            part_2b = np.exp(np.sum(np.cos(self.c * x), axis=-1) / self.n)
            return (part_1a * part_1b + part_2a.T * part_2b).T
        return ackley_grad


class Rastrigin(ObjectiveFunction):
    def __init__(
        self, 
        n: int, 
        domain: Tuple[float] = (-5.12, 5.12)
    ):
        super().__init__(n, domain=domain, minimizer=np.zeros(n))

    def get_loss_function(self) -> Callable:
        def rastrigin(x: np.ndarray) -> float:
            summand = np.square(x) - 10 * np.cos(2*np.pi * x)
            return 10*self.n + np.sum(summand, axis=-1)
        return rastrigin

    def get_grad_function(self):
        def rastrigin_grad(x: np.ndarray) -> np.ndarray[float]:
            return 2*x + 20*np.pi * np.sin(2*np.pi * x)
        return rastrigin_grad


class Rosenbrock(ObjectiveFunction):
    def __init__(
        self, 
        n: int, 
        domain: Tuple[float] = (-5., 10.), 
        a: float = 1., 
        b: float = 100.
    ):
        super().__init__(n, domain=domain, minimizer=np.ones(n))
        self.a = a
        self.b = b

    def get_loss_function(self) -> Callable:
        def rosenbrock(x: np.ndarray) -> float:
            x0 = x[..., :-1]    # Account for difference between i and i+1
            x1 = x[..., 1:]     # Slicing over the last axis
            summand_1 = self.b * np.square(x1 - np.square(x0))
            summand_2 = np.square(self.a - x0)
            return np.sum(summand_1 + summand_2, axis=-1)
        return rosenbrock

    def get_grad_function(self) -> Callable:
        def rosen_grad(x: np.ndarray) -> np.ndarray[float]:
            grad = np.zeros_like(x)
            x0 = x[..., :-1]
            x1 = x[..., 1:]

            part_1 = self.b * (x1 - np.square(x0))
            grad[..., :-1] += -4 * x0 * part_1 - 2*(self.a - x0)
            grad[..., 1:] += 2 * part_1
            return grad
        return rosen_grad


class Zakharov(ObjectiveFunction):
    def __init__(
        self, 
        n: int, 
        domain: Tuple[float] = (-5., 10.)
    ):
        super().__init__(n, domain=domain, minimizer=np.zeros(n))

    def get_loss_function(self) -> Callable:
        def zakharov(x: np.ndarray) -> float:
            i = np.arange(1, self.n+1)
            i_sum = 0.5 * (x @ i)
            x2_sum = np.sum(np.square(x), axis=-1)
            return x2_sum + np.square(i_sum) + np.power(i_sum, 4)
        return zakharov

    def get_grad_function(self) -> Callable:
        def zakharov_grad(x: np.ndarray) -> np.ndarray[float]:
            i = np.arange(1, self.n+1)
            i_sum_2 = x @ i     # The 0.5 is absorbed into the coeff calculation
            coeff = 0.5*i_sum_2 + 0.25*np.power(i_sum_2, 3)
            return (2*x + np.outer(coeff, i)).reshape(x.shape)
        return zakharov_grad

