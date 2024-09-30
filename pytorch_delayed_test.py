import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_optimizer import DelayedOptimizationWrapper
from delay_optimizer.delays.distributions import Stochastic, Decaying

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

# Define the loss function
loss_func = nn.MSELoss()

# Define the optimizer
optimizer = DelayedOptimizationWrapper(
    optim.Adam(model.parameters(), lr=0.01),
    # delay = 0,
    delay=Decaying(max_L=3, num_delays=998)
)

# Define the data
X = torch.randint(0, 2, (1001, 2))
y = (X[:, 0] ^ X[:, 1]).float()
X = X.float()

# Begin training
for i, (x, label) in enumerate(zip(X, y)):
    print("\n---------------------------------------------------------------------------------------------\n\n")

    # Train undelayed model
    optimizer.zero_grad()
    optimizer.apply_delays()
    y_hat = model(x)[0]
    loss = loss_func(y_hat, label)
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(f"Iteration: {i}, Loss: {loss.item()}")
        