import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_optimizer import DelayedOptimizationWrapper
from delay_optimizer.delays.distributions import Stochastic

# Define the XOR mdoel
class XOR(nn.Module):
    def __init__(self, input_dim=2, output_dim=1):
        super(XOR, self).__init__()
        self.lin1 = nn.Linear(input_dim, 2)
        self.lin2 = nn.Linear(2, output_dim)
    
    def forward(self, x):
        x = self.lin1(x)
        x = F.sigmoid(x)
        x = self.lin2(x)
        return x

model = XOR()

# Initialize the model weights
def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # initialize the weight tensor, here we use a normal distribution
            m.weight.data.normal_(0, 1)

weights_init(model)

# Copy the model
delayed_model = XOR()
for delayed_param, param in zip(delayed_model.parameters(), model.parameters()):
    delayed_param.data = param.data.clone()

# Define the loss function
loss_func = nn.MSELoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)
delayed_optimizer = DelayedOptimizationWrapper(
    optim.Adam(delayed_model.parameters(), lr=0.01),
    delay = 0,
    # delay=Stochastic(max_L=1, num_delays=-1)
)

# Define the data
X = torch.randint(0, 2, (10001, 2))
y = (X[:, 0] ^ X[:, 1]).float()
X = X.float()

# Begin training
for i, (x, label) in enumerate(zip(X, y)):
    # Train undelayed model
    optimizer.zero_grad()
    y_hat = model(x)[0]
    loss = loss_func(y_hat, label)
    loss.backward()
    optimizer.step()

    # Train delayed model
    delayed_optimizer.zero_grad()
    delayed_optimizer.apply_delays()
    delayed_y_hat = delayed_model(x)[0]
    delayed_loss = loss_func(delayed_y_hat, label)
    delayed_loss.backward()
    delayed_optimizer.step()

    # Check that the parameters are the same
    assert loss.item() == delayed_loss.item(), f"Losses do not match: {loss.item()} != {delayed_loss.item()}"
    for param, delayed_param in zip(model.parameters(), delayed_model.parameters()):
        assert torch.all(param.data == delayed_param.data), "Parameters do not match"

    if i % 100 == 0:
        print(f"Iteration: {i}, Loss: {loss.item()}")
        