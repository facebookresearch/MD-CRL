# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch.nn import functional as F
from models.mnist_md_autoencoder_pl import MNISTMDAutoencoderPL
from omegaconf import OmegaConf
import os
from sklearn.linear_model import LogisticRegression, LinearRegression
import utils.general as utils
log = utils.get_logger(__name__)


class MNISTMDEncodedAutoencoderPL(MNISTMDAutoencoderPL):
    def __init__(
        self,
        top_k: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)

    def on_training_start(self, *args, **kwargs):
        self.log(f"val_reconstruction_loss", 0.0)
        self.log(f"valid_penalty_loss", 0.0)
        self.log(f"val_loss", 0.0)
        return

    def training_step(self, train_batch, batch_idx):

        # x: [batch_size, z_dim]
        x, labels, domains, colors = train_batch["x"], train_batch["label"], train_batch["domain"], train_batch["color"]

        # z: [batch_size, latent_dim]
        # x_hat: [batch_size, z_dim]
        z_hat, x_hat = self(x)

        loss, reconstruction_loss, penalty_loss_value, hinge_loss_value = self.loss(x, x_hat, z_hat, domains)
        self.log(f"train_reconstruction_loss", reconstruction_loss.item())
        self.log(f"train_penalty_loss", penalty_loss_value.item())
        self.log(f"train_hinge_loss", hinge_loss_value.item())
        self.log(f"train_loss", loss.item())

        if batch_idx % 100 == 0:
            log.info(f"x.max(): {x.max()}, x_hat.max(): {x_hat.max()}, x.min(): {x.min()}, x_hat.min(): {x_hat.min()}, x.mean(): {x.mean()}, x_hat.mean(): {x_hat.mean()}, x.std(): {x.std()}, x_hat.std(): {x_hat.std()}")
            log.info(f"z_hat.max(): {z_hat.max()}, z_hat.min(): {z_hat.min()}, z_hat.mean(): {z_hat.mean()}, z_hat.std(): {z_hat.std()}")
            log.info(f"sample z_hat 1: {z_hat[50:55, 50:55]}")
            log.info(f"sample z_hat 2: {z_hat[50:55, :]}")

        self.training_step_outputs.append({"z_hat":z_hat, "labels":labels, "colors":colors})

        clf = LogisticRegression(random_state=0, max_iter=1000).fit(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), labels.detach().cpu().numpy())
        accuracy = clf.score(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), labels.detach().cpu().numpy())
        self.log(f"train_digits_accuracy_hz", accuracy, prog_bar=True)

        clf = LogisticRegression(random_state=0, max_iter=1000).fit(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), labels.detach().cpu().numpy())
        accuracy = clf.score(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), labels.detach().cpu().numpy())
        self.log(f"train_digits_accuracy_~hz", accuracy, prog_bar=True)

        if self.trainer.datamodule.train_dataset.generation_strategy == "manual": # colors are indexed
            clf = LogisticRegression(random_state=0, max_iter=1000).fit(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), colors.detach().cpu().numpy())
            accuracy = clf.score(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), colors.detach().cpu().numpy())
            self.log(f"train_colors_accuracy_hz", accuracy, prog_bar=True)
            clf = LogisticRegression(random_state=0, max_iter=1000).fit(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), colors.detach().cpu().numpy())
            accuracy = clf.score(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), colors.detach().cpu().numpy())
            self.log(f"train_colors_accuracy_~hz", accuracy, prog_bar=True)
        else: # colors are triplets of rgb, there we need to measure r2 score of linear regression
            reg = LinearRegression().fit(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), colors.detach().cpu().numpy())
            r2 = reg.score(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), colors.detach().cpu().numpy())
            self.log(f"train_colors_r2_hz", r2, prog_bar=True)
            reg = LinearRegression().fit(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), colors.detach().cpu().numpy())
            r2 = reg.score(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), colors.detach().cpu().numpy())
            self.log(f"train_colors_r2_~hz", r2, prog_bar=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):

        # images: [batch_size, num_channels, width, height]
        x, labels, domains, colors = valid_batch["x"], valid_batch["label"], valid_batch["domain"], valid_batch["color"]

        # z: [batch_size, latent_dim]
        # recons: [batch_size, num_channels, width, height]
        z_hat, x_hat = self(x)

        if batch_idx % 50 == 0:
            if self.penalty_criterion and (self.penalty_criterion["minmax"] or self.penalty_criterion["mmd"]):
                # print all z_hat mins of all domains
                log.info(f"============== z_hat min all domains ==============\n{[z_hat[(domains == i).squeeze(), :self.z_dim_invariant_model].min().detach().cpu().numpy().item() for i in range(self.num_domains)]}\n")
                # print all z_hat maxs of all domains
                log.info(f"============== z_hat max all domains ==============\n{[z_hat[(domains == i).squeeze(), :self.z_dim_invariant_model].max().detach().cpu().numpy().item() for i in range(self.num_domains)]}\n")
                log.info(f"============== ============== ============== ==============\n")
            if self.penalty_criterion and self.penalty_criterion["stddev"] == 1.:
                # print all z_hat stds of all domains for each of z_dim_invariant dimensions
                for dim in range(self.z_dim_invariant_model):
                    log.info(f"============== z_hat std all domains dim {dim} ==============\n{[z_hat[(domains == i).squeeze(), dim].std().detach().cpu().numpy().item() for i in range(self.num_domains)]}\n")
                log.info(f"============== ============== ============== ==============\n")
            if self.penalty_criterion and self.penalty_criterion["domain_classification"] == 1.:
                # log the weigth matrix of the multinomial logistic regression model
                log.info(f"============== multinomial logistic regression model weight matrix ==============\n{self.multinomial_logistic_regression.linear.weight}\n")

        loss, reconstruction_loss, penalty_loss_value, hinge_loss_value = self.loss(x, x_hat, z_hat, domains)
        self.log(f"val_reconstruction_loss", reconstruction_loss.item())
        self.log(f"val_penalty_loss", penalty_loss_value.item())
        self.log(f"val_hinge_loss", hinge_loss_value.item())
        self.log(f"val_loss", loss.item())

        self.validation_step_outputs.append({"z_hat":z_hat, "labels":labels, "colors":colors})

        clf = LogisticRegression(random_state=0, max_iter=1000).fit(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), labels.detach().cpu().numpy())
        accuracy = clf.score(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), labels.detach().cpu().numpy())
        self.log(f"digits_accuracy_hz", accuracy, prog_bar=True)

        clf = LogisticRegression(random_state=0, max_iter=1000).fit(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), labels.detach().cpu().numpy())
        accuracy = clf.score(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), labels.detach().cpu().numpy())
        self.log(f"digits_accuracy_~hz", accuracy, prog_bar=True)

        if self.trainer.datamodule.train_dataset.generation_strategy == "manual": # colors are indexed
            clf = LogisticRegression(random_state=0, max_iter=1000).fit(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), colors.detach().cpu().numpy())
            accuracy = clf.score(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), colors.detach().cpu().numpy())
            self.log(f"colors_accuracy_hz", accuracy, prog_bar=True)
            clf = LogisticRegression(random_state=0, max_iter=1000).fit(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), colors.detach().cpu().numpy())
            accuracy = clf.score(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), colors.detach().cpu().numpy())
            self.log(f"colors_accuracy_~hz", accuracy, prog_bar=True)
        else: # colors are triplets of rgb, there we need to measure r2 score of linear regression
            reg = LinearRegression().fit(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), colors.detach().cpu().numpy())
            r2 = reg.score(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), colors.detach().cpu().numpy())
            self.log(f"colors_r2_hz", r2, prog_bar=True)
            reg = LinearRegression().fit(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), colors.detach().cpu().numpy())
            r2 = reg.score(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), colors.detach().cpu().numpy())
            self.log(f"colors_r2_~hz", r2, prog_bar=True)


        return loss

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        
        z_hat = torch.cat([output["z_hat"] for output in self.validation_step_outputs], dim=0)
        labels = torch.cat([output["labels"] for output in self.validation_step_outputs], dim=0)
        colors = torch.cat([output["colors"] for output in self.validation_step_outputs], dim=0)

        # 1. predicting labels from z_hat[:z_dim_invariant_model]
        clf = LogisticRegression(random_state=0, max_iter=1000).fit(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), labels.detach().cpu().numpy())
        accuracy = clf.score(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), labels.detach().cpu().numpy())
        self.log(f"digits_accuracy_hz", accuracy, prog_bar=True)

        # 2. predicting labels from z_hat[z_dim_invariant_model:]
        clf = LogisticRegression(random_state=0, max_iter=1000).fit(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), labels.detach().cpu().numpy())
        accuracy = clf.score(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), labels.detach().cpu().numpy())
        self.log(f"digits_accuracy_~hz", accuracy, prog_bar=True)

        # 3,4. predicting colors from z_hat[:z_dim_invariant_model], and z_hat[z_dim_invariant_model:]
        if self.trainer.datamodule.train_dataset.generation_strategy == "manual": # colors are indexed
            clf = LogisticRegression(random_state=0, max_iter=1000).fit(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), colors.detach().cpu().numpy())
            accuracy = clf.score(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), colors.detach().cpu().numpy())
            self.log(f"colors_accuracy_hz", accuracy, prog_bar=True)
            clf = LogisticRegression(random_state=0, max_iter=1000).fit(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), colors.detach().cpu().numpy())
            accuracy = clf.score(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), colors.detach().cpu().numpy())
            self.log(f"colors_accuracy_~hz", accuracy, prog_bar=True)
        else: # colors are triplets of rgb, there we need to measure r2 score of linear regression
            reg = LinearRegression().fit(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), colors.detach().cpu().numpy())
            r2 = reg.score(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), colors.detach().cpu().numpy())
            self.log(f"colors_r2_hz", r2, prog_bar=True)
            reg = LinearRegression().fit(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), colors.detach().cpu().numpy())
            r2 = reg.score(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), colors.detach().cpu().numpy())
            self.log(f"colors_r2_~hz", r2, prog_bar=True)

        # overall accuracy with all z
        clf = LogisticRegression(random_state=0, max_iter=1000).fit(z_hat.detach().cpu().numpy(), labels.detach().cpu().numpy())
        accuracy = clf.score(z_hat.detach().cpu().numpy(), labels.detach().cpu().numpy())
        self.log(f"val_digits_accuracy", accuracy, prog_bar=True)

        # fit a linear regression from z to colours
        reg = LinearRegression().fit(z_hat.detach().cpu().numpy(), colors.detach().cpu().numpy())
        r2 = reg.score(z_hat.detach().cpu().numpy(), colors.detach().cpu().numpy())
        self.log(f"val_r2_colors", r2, prog_bar=True)

        # comptue the average norm of first z_dim dimensions of z
        z_norm = torch.norm(z_hat[:, :self.z_dim_invariant_model], dim=1).mean()
        self.log(f"z_norm", z_norm, prog_bar=True)
        # comptue the average norm of the last n-z_dim dimensions of z
        z_norm = torch.norm(z_hat[:, self.z_dim_invariant_model:], dim=1).mean()
        self.log(f"~z_norm", z_norm, prog_bar=True)

        self.validation_step_outputs.clear()

        return

    def on_train_start(self):

        # log the r2 scores before any training has started
        valid_dataset = torch.load(os.path.join(self.trainer.datamodule.path_to_files, f"encoded_img_multi_domain_mnist_{self.num_domains}_valid.pt"))
        
        z_hat = valid_dataset["z_hat"]
        labels = valid_dataset["label"]
        colors = valid_dataset["color"]

        clf = LogisticRegression(random_state=0, max_iter=1000).fit(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), labels.detach().cpu().numpy())
        accuracy = clf.score(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), labels.detach().cpu().numpy())
        self.log(f"val_digits_accuracy_hz_start", accuracy, prog_bar=False)
        self.log(f"digits_accuracy_hz", accuracy, prog_bar=True)

        clf = LogisticRegression(random_state=0, max_iter=1000).fit(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), labels.detach().cpu().numpy())
        accuracy = clf.score(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), labels.detach().cpu().numpy())
        self.log(f"val_digits_accuracy_~hz_start", accuracy, prog_bar=False)
        self.log(f"digits_accuracy_~hz", accuracy, prog_bar=True)

        if self.trainer.datamodule.train_dataset.generation_strategy == "manual": # colors are indexed
            clf = LogisticRegression(random_state=0, max_iter=1000).fit(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), colors.detach().cpu().numpy())
            accuracy = clf.score(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), colors.detach().cpu().numpy())
            self.log(f"val_colors_accuracy_hz_start", accuracy, prog_bar=False)
            clf = LogisticRegression(random_state=0, max_iter=1000).fit(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), colors.detach().cpu().numpy())
            accuracy = clf.score(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), colors.detach().cpu().numpy())
            self.log(f"val_colors_accuracy_~hz_start", accuracy, prog_bar=False)
        else: # colors are triplets of rgb, there we need to measure r2 score of linear regression
            reg = LinearRegression().fit(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), colors.detach().cpu().numpy())
            r2 = reg.score(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), colors.detach().cpu().numpy())
            self.log(f"val_colors_r2_hz_start", r2, prog_bar=False)
            self.log(f"colors_r2_hz", r2, prog_bar=False)
            reg = LinearRegression().fit(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), colors.detach().cpu().numpy())
            r2 = reg.score(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), colors.detach().cpu().numpy())
            self.log(f"val_colors_r2_~hz_start", r2, prog_bar=False)
            self.log(f"colors_r2_~hz", r2, prog_bar=False)

        # overall accuracy with all z
        clf = LogisticRegression(random_state=0, max_iter=1000).fit(z_hat.detach().cpu().numpy(), labels.detach().cpu().numpy())
        accuracy = clf.score(z_hat.detach().cpu().numpy(), labels.detach().cpu().numpy())
        self.log(f"val_digits_accuracy_start", accuracy, prog_bar=False)

        return

