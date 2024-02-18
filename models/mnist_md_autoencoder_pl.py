# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch.nn import functional as F
from models.base_pl import BasePl
import hydra
from omegaconf import OmegaConf
import wandb
import os
from utils.disentanglement_utils import linear_disentanglement, permutation_disentanglement
import utils.general as utils
log = utils.get_logger(__name__)
from models.utils import penalty_loss_minmax, penalty_loss_stddev, hinge_loss


class MNISTMDAutoencoderPL(BasePl):
    def __init__(
        self,
        top_k: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)

        self.penalty_weight = self.hparams.get("penalty_weight", 1.0)
        self.model = hydra.utils.instantiate(self.hparams.autoencoder, _recursive_=False)
        self.save_encoded_data = self.hparams.get("save_encoded_data", False)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.loaded_img_data = False

    def on_training_start(self, *args, **kwargs):
        self.log(f"val_reconstruction_loss", 0.0)
        self.log(f"valid_penalty_loss", 0.0)
        self.log(f"val_loss", 0.0)
        return

    def forward(self, x):
        from models.modules.mlp_ae import FCAE
        if isinstance(self.model, FCAE):
            if len(x.size()) > 2:
                x = x.view(x.size(0), -1) # flatten the input
                z, x_hat = self.model(x)
                x_hat = x_hat.view(x_hat.size(0), 3, 28, 28) # reshape the output
            else:
                z, x_hat = self.model(x)
            return z, x_hat
        else:
            return self.model(x)

    def training_step(self, train_batch, batch_idx):

        # images: [batch_size, num_channels, width, height]
        images, labels, domains, colors = train_batch["image"], train_batch["label"], train_batch["domain"], train_batch["color"]

        # z: [batch_size, latent_dim]
        # recons: [batch_size, num_channels, width, height]
        z, recons = self(images)

        loss, reconstruction_loss, penalty_loss_value, hinge_loss_value = self.loss(images, recons, z, domains)
        self.log(f"train_reconstruction_loss", reconstruction_loss.item())
        self.log(f"train_penalty_loss", penalty_loss_value.item())
        self.log(f"train_loss", loss.item())

        if batch_idx % 100 == 0:
            log.info(f"images.max(): {images.max()}, recons.max(): {recons.max()}, images.min(): {images.min()}, recons.min(): {recons.min()}\n images.mean(): {images.mean()}, recons.mean(): {recons.mean()}, images.std(): {images.std()}, recons.std(): {recons.std()}")
            log.info(f"z_hat.max(): {z.max()}, z_hat.min(): {z.min()}, z_hat.mean(): {z.mean()}, z_hat.std(): {z.std()}")

        if self.save_encoded_data:
            self.training_step_outputs.append({"z_hat":z})

        return loss

    def validation_step(self, valid_batch, batch_idx):

        # images: [batch_size, num_channels, width, height]
        images, labels, domains, colors = valid_batch["image"], valid_batch["label"], valid_batch["domain"], valid_batch["color"]

        # z: [batch_size, latent_dim]
        # recons: [batch_size, num_channels, width, height]
        z, recons = self(images)

        # we have the set of labels and latents. We want to train a classifier to predict the labels from latents
        # using multinomial logistic regression using sklearn
        # import sklearn
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.metrics import accuracy_score, r2_score
        # fit a multinomial logistic regression from z to labels and 
        # multinomial/linear regression to colors (based on color representations)

        # 1. predicting labels from z[:z_dim_invariant_model]
        clf = LogisticRegression(random_state=0, max_iter=1000).fit(z[:, :self.z_dim_invariant_model].detach().cpu().numpy(), labels.detach().cpu().numpy())
        accuracy = clf.score(z[:, :self.z_dim_invariant_model].detach().cpu().numpy(), labels.detach().cpu().numpy())
        self.log(f"digits_accuracy_z", accuracy, prog_bar=True)

        # 2. predicting colors from z[:z_dim_invariant_model]
        if self.trainer.datamodule.train_dataset.generation_strategy == "manual": # colors are indexed, therefore we require classification
            clf = LogisticRegression(random_state=0, max_iter=1000).fit(z[:, :self.z_dim_invariant_model].detach().cpu().numpy(), colors.detach().cpu().numpy())
            pred_colors = clf.predict(z[:, :self.z_dim_invariant_model].detach().cpu().numpy())
            accuracy = accuracy_score(colors.detach().cpu().numpy(), pred_colors)
            self.log(f"colors_accuracy_z", accuracy, prog_bar=True)
        else: # colors are triplets of rgb, there we need to measure r2 score of linear regression
            reg = LinearRegression().fit(z[:, :self.z_dim_invariant_model].detach().cpu().numpy(), colors.detach().cpu().numpy())
            r2 = reg.score(z[:, :self.z_dim_invariant_model].detach().cpu().numpy(), colors.detach().cpu().numpy())
            self.log(f"colors_r2_z", r2, prog_bar=True)

        # 3. predicting labels from z[z_dim_invariant_model:]
        clf = LogisticRegression(random_state=0, max_iter=1000).fit(z[:, self.z_dim_invariant_model:].detach().cpu().numpy(), labels.detach().cpu().numpy())
        accuracy = clf.score(z[:, self.z_dim_invariant_model:].detach().cpu().numpy(), labels.detach().cpu().numpy())
        self.log(f"digits_accuracy_~z", accuracy, prog_bar=True)

        # 4. predicting colors from z[z_dim_invariant_model:]
        if self.trainer.datamodule.train_dataset.generation_strategy == "manual":
            clf = LogisticRegression(random_state=0, max_iter=1000).fit(z[:, self.z_dim_invariant_model:].detach().cpu().numpy(), colors.detach().cpu().numpy())
            pred_colors = clf.predict(z[:, self.z_dim_invariant_model:].detach().cpu().numpy())
            accuracy = accuracy_score(colors.detach().cpu().numpy(), pred_colors)
            self.log(f"colors_accuracy_~z", accuracy, prog_bar=True)
        else:
            reg = LinearRegression().fit(z[:, self.z_dim_invariant_model:].detach().cpu().numpy(), colors.detach().cpu().numpy())
            r2 = reg.score(z[:, self.z_dim_invariant_model:].detach().cpu().numpy(), colors.detach().cpu().numpy())
            self.log(f"colors_r2_~z", r2, prog_bar=True)

        # overall accuracy with all z
        clf = LogisticRegression(random_state=0, max_iter=1000).fit(z.detach().cpu().numpy(), labels.detach().cpu().numpy())
        accuracy = clf.score(z.detach().cpu().numpy(), labels.detach().cpu().numpy())
        self.log(f"val_digits_accuracy", accuracy, prog_bar=True)

        # fit a linear regression from z to colours
        reg = LinearRegression().fit(z.detach().cpu().numpy(), colors.detach().cpu().numpy())
        r2 = reg.score(z.detach().cpu().numpy(), colors.detach().cpu().numpy())
        self.log(f"val_r2_colors", r2, prog_bar=True)

        # comptue the average norm of first z_dim dimensions of z
        z_norm = torch.norm(z[:, :self.z_dim_invariant_model], dim=1).mean()
        self.log(f"z_norm", z_norm, prog_bar=True)
        # comptue the average norm of the last n-z_dim dimensions of z
        z_norm = torch.norm(z[:, self.z_dim_invariant_model:], dim=1).mean()
        self.log(f"~z_norm", z_norm, prog_bar=True)


        loss, reconstruction_loss, penalty_loss_value, hinge_loss_value = self.loss(images, recons, z, domains)
        self.log(f"val_reconstruction_loss", reconstruction_loss.item())
        self.log(f"val_penalty_loss", penalty_loss_value.item())
        self.log(f"val_loss", loss.item())

        if self.save_encoded_data:
            self.validation_step_outputs.append({"z_hat":z})

        return loss

    def on_train_epoch_end(self):

        if self.save_encoded_data:
            # load the train dataset, and replace its "image" key with the new_data["z_hat"] key
            # and save it as a pt file
            new_data = dict.fromkeys(self.train_dataset.data[0].keys())

            # stack the values of each key in the new_data dict
            # new_data is a list of dicts
            for key in new_data.keys():
                new_data[key] = torch.stack([torch.tensor(self.train_dataset.data[i][key]) for i in range(len(self.train_dataset))], dim=0)
            new_data.pop("image", None)
            for key in self.training_step_outputs[0].keys():
                new_data[key] = torch.zeros((len(self.train_dataset), self.training_step_outputs[0][key].shape[-1]))

            for batch_idx, training_step_output in enumerate(self.training_step_outputs):
                # save the outputs of the encoder as a new dataset
                training_step_output_batch_size = list(training_step_output.values())[0].shape[0]
                start = batch_idx * self.trainer.datamodule.train_dataloader().batch_size
                end = start + min(self.trainer.datamodule.train_dataloader().batch_size, training_step_output_batch_size)
                for key, val in zip(training_step_output.keys(), training_step_output.values()):
                    try:
                        new_data[key][start:end] = val.detach().cpu()
                    except:
                        new_data[key][start:end] = val.unsqueeze(-1).detach().cpu()

            # save the new dataset as a pt file in hydra run dir or working directory
            log.info(f"Saving the encoded training dataset of length {len(new_data[key])} at: {os.getcwd()}")
            torch.save(new_data, os.path.join(os.getcwd(), f"encoded_img_{self.trainer.datamodule.datamodule_name}_train.pt"))
            new_data = []

        self.training_step_outputs.clear()

        return
    
    def on_validation_epoch_end(self):
        # import code
        # code.interact(local=locals())
        if self.save_encoded_data:
            # load the valid dataset, and replace its "image" key with the new_data["z_hat"] key
            # and save it as a pt file
            if not self.loaded_img_data:
                path = os.path.join(self.trainer.datamodule.path_to_files, f"valid_dataset_{self.trainer.datamodule.datamodule_name}_{self.trainer.datamodule.num_samples['valid']}.pt")
                self.valid_dataset = torch.load(path)
                log.info(f"Loaded the validation dataset of length {len(self.valid_dataset)} from: {path}")
                path = os.path.join(self.trainer.datamodule.path_to_files, f"train_dataset_{self.trainer.datamodule.datamodule_name}_{self.trainer.datamodule.num_samples['train']}.pt")
                self.train_dataset = torch.load(path)
                log.info(f"Loaded the training dataset of length {len(self.train_dataset)} from: {path}")
                self.loaded_img_data = True
            
            new_data = dict.fromkeys(self.valid_dataset.data[0].keys())

            # stack the values of each key in the new_data dict
            # new_data is a list of dicts
            for key in new_data.keys():
                new_data[key] = torch.stack([torch.tensor(self.valid_dataset.data[i][key]) for i in range(len(self.valid_dataset))], dim=0)
            new_data.pop("image", None)
            for key in self.validation_step_outputs[0].keys():
                new_data[key] = torch.zeros((len(self.valid_dataset), self.validation_step_outputs[0][key].shape[-1]))

            for batch_idx, validation_step_output in enumerate(self.validation_step_outputs):
                # save the outputs of the encoder as a new dataset
                validation_step_output_batch_size = list(validation_step_output.values())[0].shape[0]
                start = batch_idx * self.trainer.datamodule.val_dataloader().batch_size
                end = start + min(self.trainer.datamodule.val_dataloader().batch_size, validation_step_output_batch_size)
                for key, val in zip(validation_step_output.keys(), validation_step_output.values()):
                    try:
                        new_data[key][start:end] = val.detach().cpu()
                    except:
                        new_data[key][start:end] = val.unsqueeze(-1).detach().cpu()
            
            # save the new dataset as a pt file in hydra run dir or working directory
            log.info(f"Saving the encoded validation dataset of length {len(new_data[key])} at: {os.getcwd()}")
            torch.save(new_data, os.path.join(os.getcwd(), f"encoded_img_{self.trainer.datamodule.datamodule_name}_valid.pt"))

            valid_dataset = []
            new_data = []
        self.validation_step_outputs.clear()

        return
