import torch
from torch import nn
from torch.nn import functional as F
import hydra
from typing import List, Callable, Union, Any, TypeVar, Tuple

class BetaVAE(nn.Module):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 latent_dim: int,
                 beta: int = 4,
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = torch.nn.Sequential(
            *[hydra.utils.instantiate(layer_config) for _, layer_config in kwargs['encoder_layers'].items()]
        )

        self.decoder = torch.nn.Sequential(
            *[hydra.utils.instantiate(layer_config) for _, layer_config in kwargs['decoder_layers'].items()]
        )


    def encode(self, x):

        z_mean, z_logvar = self.encoder(x).chunk(2, dim=-1)
        z_sample = self.reparameterize(z_mean, z_logvar)
        return z_sample, z_mean, z_logvar

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x, **kwargs):

        z_hat, z_mean, z_logvar = self.encode(x)
        x_hat = self.decode(z_hat)

        return x_hat, z_hat, z_mean, z_logvar

    def loss(self, x, x_hat, z_mean, z_logvar):

        recon_loss = torch.mean( ((x-x_hat)**2), dim=1 )
        kl = torch.sum(0.5 * ( (z_mean)**2 + torch.exp(z_logvar) - z_logvar - 1 ), dim=1)

        loss = torch.mean(recon_loss + self.beta * kl)
        return loss, torch.mean(recon_loss), torch.mean(kl)
