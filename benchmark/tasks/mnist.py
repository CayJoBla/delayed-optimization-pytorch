from torch import nn

from .task import OptimizationTask
from .modules import mnist_2c2d_net
from ..datasets import MNIST


class mnist_task(OptimizationTask):
    def __init__(self):
        super().__init__()
        self.data = MNIST()
        self.loss_func = nn.CrossEntropyLoss()

class mnist_2c2d(mnist_task):
    def __init__(self, num_outputs=10):
        super().__init__()
        self.model = mnist_2c2d_net(num_outputs).to(self.device)

