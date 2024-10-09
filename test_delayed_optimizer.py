import os
import unittest
import torch
from torch import nn

from delay_optimizer.delays import DelayedOptimizer
from delay_optimizer.delays.distributions import Stochastic


class XOR(nn.Module):
    def __init__(self, input_dim=2, output_dim=1):
        super(XOR, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2)
        self.fc2 = nn.Linear(2, output_dim)
        self.relu = nn.ReLU()
        self.loss_func = nn.MSELoss()
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def get_test_model(filename="test_params.pt"):
    if filename in os.listdir():
        model = XOR()
        model.load_state_dict(torch.load(filename, weights_only=True))
    else:
        model = XOR()
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
        torch.save(model.state_dict(), filename)
    return model

def get_test_data():
    X = torch.randint(0, 2, (1000, 2))
    y = (X[:, 0] ^ X[:, 1]).float()
    X = X.float()
    return X, y


class TestDelayedOptimizerPytorch(unittest.TestCase):
    base_optimizer = torch.optim.Adam

    def test_undelayed(self):
        undelayed_model = get_test_model()
        undelayed_optimizer = DelayedOptimizer(
            params = undelayed_model.parameters(),
            optimizer_class = self.base_optimizer,
            delay = 0,
        )

        vanilla_model = get_test_model()
        vanilla_optimizer = self.base_optimizer(vanilla_model.parameters())

        X, labels = get_test_data()
        for x, label in zip(X, labels):
            undelayed_optimizer.zero_grad()
            undelayed_optimizer.apply_delays()
            y_hat = undelayed_model(x).squeeze()
            loss = undelayed_model.loss_func(y_hat, label)
            loss.backward()

            vanilla_optimizer.zero_grad()
            y_hat = vanilla_model(x).squeeze()
            loss = vanilla_model.loss_func(y_hat, label)
            loss.backward()

            for p1, p2 in zip(undelayed_model.parameters(), 
                    vanilla_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))

    def test_stochastic_delay(self):
        model = get_test_model()
        delayed_optimizer = DelayedOptimizer(
            params = model.parameters(),
            optimizer_class = self.base_optimizer,
            delay = Stochastic(max_L=5),
        )

        X, labels = get_test_data()
        for x, label in zip(X, labels):
            delayed_optimizer.zero_grad()
            delayed_optimizer.apply_delays()
            y_hat = model(x).squeeze()
            loss = model.loss_func(y_hat, label)
            loss.backward()

        # TODO: Should check whether delays are applied properly
            
    def test_mixed_delay(self):
        # TODO: Implement this
        return


if __name__ == '__main__':
    unittest.main()
