import torch
from torch.nn import functional as F
from .base_pl import BasePl
import hydra
from omegaconf import OmegaConf
import wandb
import os
import numpy as np
import utils.general as utils
log = utils.get_logger(__name__)

class BetaVAEPL(BasePl):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)

        self.z_dim = self.hparams.get("z_dim", 128)
        self.z_dim_invariant_data = self.hparams.get("z_dim_invariant_data", 2)
        self.model = hydra.utils.instantiate(self.hparams.autoencoder, _recursive_=False)
        self.mu = torch.nn.Linear(self.z_dim, self.z_dim)
        self.logvar = torch.nn.Linear(self.z_dim, self.z_dim)
        print(f"self.mu:\n{self.mu}\nself.logvar:\n{self.logvar}")
        print(f"self.model:\n{self.model}")
        self.beta = self.hparams.get("beta", 1)
        # select half of z_dim dimensions randomly
        self.inv_indices = np.random.choice(self.z_dim, self.z_dim//2, replace=False)
        self.spu_indices = np.setdiff1d(np.arange(self.z_dim), self.inv_indices)
        self.save_encoded_data = self.hparams.get("save_encoded_data", False)
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x, **kwargs):

        z_hat, z_mean, z_logvar = self.encode(x)
        x_hat = self.decode(z_hat)

        return x_hat, z_hat, z_mean, z_logvar
    
    def encode(self, x):

        z_hat = self.model.encoder(x)
        z_mean, z_logvar = self.mu(z_hat), self.logvar(z_hat)
        return z_hat, z_mean, z_logvar

    def decode(self, z):
        x_hat = self.model.decoder(z)
        return x_hat

    # def forward(self, x, **kwargs):

    #     z_hat, z_mean, z_logvar = self.encode(x)
    #     x_hat = self.decode(z_hat)

    #     return x_hat, z_hat, z_mean, z_logvar
    
    # def encode(self, x):

    #     z_mean, z_logvar = self.mu(self.model.encoder(x)), self.logvar(self.model.encoder(x))
    #     z_sample = self.reparameterize(z_mean, z_logvar)
    #     return z_sample, z_mean, z_logvar

    # def decode(self, z):
    #     x_hat = self.model.decoder(z)
    #     return x_hat

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss(self, x, x_hat, z_mean, z_logvar):

        recon_loss = F.mse_loss(x_hat, x, reduction="mean") # torch.mean( ((x-x_hat)**2), dim=1 )
        kl = torch.sum(0.5 * ( (z_mean)**2 + torch.exp(z_logvar) - z_logvar - 1 ), dim=1)

        loss = recon_loss + self.beta * torch.mean(kl)
        return loss, recon_loss, torch.mean(kl)

    def on_training_start(self, *args, **kwargs):
        self.log(f"val_reconstruction_loss", 0.0)
        self.log(f"val_kl_loss", 0.0)
        self.log(f"val_loss", 0.0)
        return

    def training_step(self, train_batch, batch_idx):

        x = train_batch["x"]

        # z: [batch_size, latent_dim]
        x_hat, z_hat, mu, logvar = self(x)
        loss, recon_loss, kl = self.loss(x, x_hat, mu, logvar)
        self.log(f"train_loss", loss.item(), prog_bar=True)
        self.log(f"train_recon_loss", recon_loss.item(), prog_bar=True)
        self.log(f"train_kl_loss", kl.item(), prog_bar=True)

        if batch_idx % 20 == 0:
            log.info(f"x.max(): {x.max()}, x_hat.max(): {x_hat.max()}, x.min(): {x.min()}, x_hat.min(): {x_hat.min()}, x.mean(): {x.mean()}, x_hat.mean(): {x_hat.mean()}")
            log.info(f"z_hat.max(): {z_hat.max()}, z_hat.min(): {z_hat.min()}, z_hat.mean(): {z_hat.mean()}, z_hat.std(): {z_hat.std()}")
            log.info(f"z_hat[40:50, 10:20]: {z_hat[40:50, 10:20]}")
            # get the covariance matrix of z_hat
            z_hat_cov = np.cov(z_hat.detach().cpu().numpy().T)
            print(f"z_hat_cov:\n{z_hat_cov}")
            print(f"z_hat eigen values:\n{np.linalg.eigvals(z_hat_cov)}")
            # log the rank of the encoder
            print(f"encoder_rank:\n{np.linalg.matrix_rank(self.model.encoder.layers[0].weight.detach().cpu().numpy())}")
            # log the eigenvalues as well
            u, s, v = np.linalg.svd(self.model.encoder.layers[0].weight.detach().cpu().numpy())
            print(f"encoder singular values:\n{s}")

        if self.save_encoded_data:
            self.training_step_outputs.append({"z_hat":z_hat})

        return loss

    def validation_step(self, valid_batch, batch_idx):

        x = valid_batch["x"]

        # z: [batch_size, latent_dim]
        x_hat, z_hat, mu, logvar = self(x)
        loss, recon_loss, kl = self.loss(x, x_hat, mu, logvar)
        self.log(f"val_loss", loss.item(), prog_bar=True)
        self.log(f"val_recons_loss", recon_loss.item(), prog_bar=False)
        self.log(f"val_kl_loss", kl.item(), prog_bar=False)

        z = valid_batch["z"]
        r2, _ = self.compute_r2(x, z)
        self.log(f"x_z_r2", r2, prog_bar=True)
        # fit a linear regression from z_hat on z
        r2, _ = self.compute_r2(z, z_hat)
        self.log(f"r2", r2, prog_bar=True)
        r2, _ = self.compute_r2(z_hat, z)
        self.log(f"~r2", r2, prog_bar=True)

        # fit a linear regression from z_hat on z_invariant dimensions
        # try:
        #     z_invariant = valid_batch["z_invariant"] # [batch_size, n_balls_invariant * z_dim_ball]
        # except KeyError:
        z_invariant = z[:, :self.z_dim_invariant_data]
        r2, _ = self.compute_r2(z_invariant, z_hat[:, self.inv_indices])
        self.log(f"hz_z_r2", r2, prog_bar=False)
        r2, _ = self.compute_r2(z_hat[:, self.inv_indices], z_invariant)
        self.log(f"hz_z_~r2", r2, prog_bar=True)
        
        # fit a linear regression from z_hat on z_spurious dimensions
        # try:
        #     z_spurious = valid_batch["z_spurious"] # [batch_size, n_balls_spurious * z_dim_ball]
        # except KeyError:
        z_spurious = z[:, self.z_dim_invariant_data:]
        r2, _ = self.compute_r2(z_spurious, z_hat[:, self.spu_indices])
        self.log(f"hz_~z_r2", r2, prog_bar=False)
        r2, _ = self.compute_r2(z_hat[:, self.spu_indices], z_spurious)
        self.log(f"hz_~z_~r2", r2, prog_bar=True)

        if self.save_encoded_data:
            self.validation_step_outputs.append({"z_hat":z_hat})

        return loss

    def on_train_epoch_end(self):

        self.training_step_outputs.clear()
        return
    
    def on_validation_epoch_end(self):

        self.validation_step_outputs.clear()
        return

    def on_train_start(self):

        # log the r2 scores before any training has started
        valid_dataset = self.trainer.datamodule.valid_dataset
        from datamodule.md_balls_encoded_dataset import BallsMultiDomainEncodedDataset
        from datamodule.md_mixing_encoded_dataset import MixingMultiDomainEncodedDataset

        if isinstance(valid_dataset, BallsMultiDomainEncodedDataset) or isinstance(valid_dataset, MixingMultiDomainEncodedDataset):
            z = torch.stack([t["z"] for t in valid_dataset], dim=0)
            x = torch.stack([t["x"] for t in valid_dataset], dim=0)
        else:
            z = torch.stack([t["z"] for t in valid_dataset.data], dim=0)
            x = torch.stack([t["x"] for t in valid_dataset.data], dim=0)
            # z = valid_dataset["z"]
            # x = valid_dataset["x"]

        _, z_hat, _, _ = self(x)

        z_invariant = z[:, :self.z_dim_invariant_data]
        z_spurious = z[:, self.z_dim_invariant_data:]

        r2, _ = self.compute_r2(z, z_hat)
        self.log(f"r2_linreg_start", r2, prog_bar=False)
        r2, _ = self.compute_r2(z_hat, z)
        self.log(f"~r2_linreg_start", r2, prog_bar=False)

        # fit a linear regression from z_hat invariant dims to z_invariant dimensions
        # z_invariant: [batch_size, n_balls_invariant * z_dim_ball]
        r2, _ = self.compute_r2(z_invariant, z_hat[:, :self.z_dim_invariant_model])
        self.log(f"hz_z_r2_linreg_start", r2, prog_bar=False)
        r2, _ = self.compute_r2(z_hat[:, :self.z_dim_invariant_model], z_invariant)
        self.log(f"hz_z_~r2_linreg_start", r2, prog_bar=False)

        # fit a linear regression from z_hat invariant dims to z_spurious dimensions
        # z_spurious: [batch_size, n_balls_spurious * z_dim_ball]
        r2, _ = self.compute_r2(z_spurious, z_hat[:, :self.z_dim_invariant_model])
        self.log(f"hz_~z_r2_linreg_start", r2, prog_bar=False)
        r2, _ = self.compute_r2(z_hat[:, :self.z_dim_invariant_model], z_spurious)
        self.log(f"hz_~z_~r2_linreg_start", r2, prog_bar=False)
