from torch import nn

from .modules import mnist_2c2d_net
from ..datasets import MNIST


class MNISTTask:
    def __init__(self):
        self.data = MNIST()
        self.loss_func = nn.CrossEntropyLoss()

class mnist_2c2d(MNISTTask):
    def __init__(self, num_outputs=10):
        super().__init__()
        self.model = mnist_2c2d_net(num_outputs)

