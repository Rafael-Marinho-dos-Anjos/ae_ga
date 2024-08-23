
import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 4, 7),
            nn.Tanh(),
            nn.Conv2d(4, 8, 7),
            nn.Tanh(),
            nn.Conv2d(8, 8, 5),
            nn.Tanh(),
            nn.Conv2d(8, 8, 5),
            nn.Tanh(),
            nn.Conv2d(8, 16, 5),
            nn.Tanh(),
            nn.Conv2d(16, 16, 3),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 3),
            nn.Sigmoid(),
            nn.ConvTranspose2d(16, 8, 5),
            nn.Sigmoid(),
            nn.ConvTranspose2d(8, 8, 5),
            nn.Sigmoid(),
            nn.ConvTranspose2d(8, 8, 5),
            nn.Sigmoid(),
            nn.ConvTranspose2d(8, 4, 7),
            nn.Sigmoid(),
            nn.ConvTranspose2d(4, n_channels, 7)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def deep_feat(self, x):
        return self.encoder(x)

if __name__ == "__main__":
    model = AutoEncoder(1)
    print(model(torch.ones((16, 1, 28, 28))).shape)
    print(model.deep_feat(torch.ones((16, 1, 28, 28))).shape)
