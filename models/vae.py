import torch
import torch.nn as nn
from models.ae import IMG_DIM, make_decoder


class VAE(nn.Module):
    def __init__(self, latent_dim=64, hidden=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(IMG_DIM, hidden), nn.ReLU())
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.decoder = make_decoder(latent_dim, hidden)

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        """z = mu + sigma * eps, eps ~ N(0, I): keeps sampling differentiable."""
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    def reconstruct(self, x):
        """Deterministic reconstruction (z = mu), used for test metrics."""
        return self.decode(self.encode(x)[0])

    @torch.no_grad()
    def sample(self, n, device):
        return self.decode(torch.randn(n, self.latent_dim, device=device))
