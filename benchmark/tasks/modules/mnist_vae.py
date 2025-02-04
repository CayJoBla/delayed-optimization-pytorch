from torch import nn

class mnist_vae_net(nn.Module):
    """A basic VAE for MNIST."""

    def __init__(self, num_latent=8):
        super().__init__()
        self.num_latent = num_latent

        # TODO: stride > 1 does not work with padding="same"

        # Encoder
        self.conv1 = nn.Sequential([
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=0,
            )
        ])
        self.dropout1 = nn.Dropout(p=0.2)

        self.conv2 = nn.Sequential([
            nn.ZeroPad2d((1, 2, 1, 2)),  # To mimic padding="same"
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=0,
            ),
        ])
        self.dropout2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Sequential([
            nn.ZeroPad2d((1, 2, 1, 2)),  # To mimic padding="same"
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
        ])
        self.dropout3 = nn.Dropout(p=0.2)

        self.dense1 = nn.Linear(in_features=7*7*64, out_features=self.num_latent)
        self.dense2 = nn.Linear(in_features=7*7*64, out_features=self.num_latent)

        # Decoder
        self.dense3 = nn.Linear(in_features=self.num_latent, out_features=24)
        self.dense4 = nn.Linear(in_features=24, out_features=24*2+1)

        self.deconv1 = nn.ConvTranspose2d(
            in_channels=1,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding="same",
        )
        self.dropout4 = nn.Dropout(p=0.2)

        self.deconv2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=64,
            kernel_size=4,
            stride=1,
            padding="same",
        )
        self.dropout5 = nn.Dropout(p=0.2)

        self.deconv3 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=64,
            kernel_size=4,
            stride=1,
            padding="same",
        )
        self.dropout6 = nn.Dropout(p=0.2)

        self.dense5 = nn.Linear(in_features=14*14*64, out_features=28*28)

        # Initialize weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.ConvTranspose2d):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_uniform_(module.weight)

    def encode(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.3) # 14x14x64
        x = self.dropout1(x)

        x = F.leaky_relu(self.conv2(x), negative_slope=0.3) # 7x7x64
        x = self.dropout2(x)

        x = F.leaky_relu(self.conv3(x), negative_slope=0.3) # 7x7x64    
        x = self.dropout3(x)

        x = x.view(-1, 7*7*64)  # 7*7*64

        mean = self.dense1(x)           # num_latent
        std_dev = 0.5 * self.dense2(x)  # num_latent
        eps = torch.randn_like(std_dev)
        z = mean + eps * torch.exp(std_dev) # num_latent

        return z, mean, std_dev

    def decode(self, z):
        x = F.leaky_relu(self.dense3(z), negative_slope=0.3)    # 24
        x = F.leaky_relu(self.dense4(x), negative_slope=0.3)    # 49

        x = x.view(-1, 1, 7, 7) # 7x7x1

        x = F.relu(self.deconv1(x)) # 14x14x64
        x = self.dropout4(x)

        x = F.relu(self.deconv2(x)) # 14x14x64
        x = self.dropout5(x)

        x = F.relu(self.deconv3(x)) # 14x14x64
        x = self.dropout6(x)

        x = x.view(-1, 14 * 14 * 64)    # 14*14*64

        x = F.sigmoid(self.dense5(x))   # 28*28

        images = x.view(-1, 1, 28, 28)  # 28x28x1

        return images

    def forward(self, x):
        z, mean, std_dev = self.encode(x)

        image = self.decode(z)

        return image, mean, std_dev