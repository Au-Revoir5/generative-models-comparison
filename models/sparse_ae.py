import torch.nn as nn
from models.ae import IMG_DIM, make_decoder


class SparseAE(nn.Module):
    def __init__(self, latent_dim=64, hidden=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_hidden = nn.Sequential(nn.Flatten(), nn.Linear(IMG_DIM, hidden), nn.ReLU())
        self.enc_latent = nn.Linear(hidden, latent_dim)
        self.decoder = make_decoder(latent_dim, hidden)

    def hidden(self, x):
        return self.enc_hidden(x)                 # the sparse code h (256-dim)

    def encode(self, x):
        return self.enc_latent(self.hidden(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        h = self.hidden(x)
        return self.decode(self.enc_latent(h)), h
