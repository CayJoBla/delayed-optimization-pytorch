from torch import nn

class mnist_mlp_net(nn.Sequential):
    """A basic MLP architecture for MNIST."""

    def __init__(self, num_outputs):
        super().__init__()

        self.num_inputs = 784
        self.add_module("flatten", nn.Flatten())
        self.add_module("dense1", nn.Linear(self.num_inputs, 1000))
        self.add_module("relu1", nn.ReLU())
        self.add_module("dense2", nn.Linear(1000, 500))
        self.add_module("relu2", nn.ReLU())
        self.add_module("dense3", nn.Linear(500, 100))
        self.add_module("relu3", nn.ReLU())
        self.add_module("dense4", nn.Linear(100, num_outputs))
        self.num_outputs = num_outputs

        # Init weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)
                stddev = 3e-2
                module.weight.data = nn.init.trunc_normal_(
                    module.weight.data, 
                    mean=0, 
                    std=stddev,
                    a=-2*stddev,
                    b=2*stddev,
                )
