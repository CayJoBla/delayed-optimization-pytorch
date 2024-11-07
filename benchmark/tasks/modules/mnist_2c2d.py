from torch import nn

class mnist_2c2d_net(nn.Sequential):
    def __init__(self, num_outputs):
        super().__init__()
        self.add_module(
            "conv1",
            nn.Conv2d(          # Input: 28x28x1, Output: 28x28x32
                in_channels=1,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding='same',
            ),
        )
        self.add_module("relu1", nn.ReLU())
        self.add_module(        # Input: 28x28x32, Output: 14x14x32
            "max_pool1", 
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                padding=0,
            ),
        )

        self.add_module(        # Input: 14x14x32, Output: 14x14x64
            "conv2",
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding='same',
            ),
        )
        self.add_module("relu2", nn.ReLU())
        self.add_module(        # Input: 14x14x64, Output: 7x7x64
            "max_pool2", 
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                padding=0,
            ),
        )

        self.add_module("flatten", nn.Flatten())    # Output: 7*7*64

        self.add_module(        # Input: 7*7*64, Output: 1024
            "dense1", 
            nn.Linear(in_features=7 * 7 * 64, out_features=1024)
        )
        self.add_module("relu3", nn.ReLU())

        self.add_module(        # Input: 1024, Output: num_outputs
            "dense2", 
            nn.Linear(in_features=1024, out_features=num_outputs)
        )

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.05)
                stddev = 0.05
                module.weight.data = nn.init.trunc_normal_(
                    module.weight.data, 
                    mean=0, 
                    std=stddev, 
                    a=-2*stddev, 
                    b=2*stddev,
                )

