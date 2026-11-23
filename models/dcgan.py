import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim=100, ngf=64):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            # z: (B, z_dim, 1, 1) -> (B, 4ngf, 7, 7)
            nn.ConvTranspose2d(z_dim, ngf * 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            # -> (B, 2ngf, 14, 14)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            # -> (B, ngf, 28, 28)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            # -> (B, 1, 28, 28)
            nn.Conv2d(ngf, 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), self.z_dim, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            # (B, 1, 28, 28) -> (B, ndf, 14, 14)
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (B, 2ndf, 7, 7)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),
            # -> (B, 1, 1, 1) logit
            nn.Conv2d(ndf * 2, 1, 7, 1, 0),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
