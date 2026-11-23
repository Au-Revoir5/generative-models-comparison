import torch.nn as nn

IMG_DIM = 28 * 28


def make_encoder(latent_dim=64, hidden=256):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(IMG_DIM, hidden), nn.ReLU(),
        nn.Linear(hidden, latent_dim),
    )


def make_decoder(latent_dim=64, hidden=256):
    """Shared decoder. The MLP frozen-latent decoder (Stage 4) reuses this exact
    function, so the MLP-vs-AE comparison is fair."""
    return nn.Sequential(
        nn.Linear(latent_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, IMG_DIM), nn.Sigmoid(),
        nn.Unflatten(1, (1, 28, 28)),
    )


class VanillaAE(nn.Module):
    def __init__(self, latent_dim=64, hidden=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = make_encoder(latent_dim, hidden)
        self.decoder = make_decoder(latent_dim, hidden)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))
