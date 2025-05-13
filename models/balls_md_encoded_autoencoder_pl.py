# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch.nn import functional as F
from .balls_autoencoder_pl import BallsAutoencoderPL
import hydra
from omegaconf import OmegaConf
import wandb
import os
from utils.disentanglement_utils import linear_disentanglement, permutation_disentanglement
import utils.general as utils
log = utils.get_logger(__name__)
from models.utils import penalty_loss_minmax, penalty_loss_stddev, penalty_domain_classification, hinge_loss, mmd_loss


class BallsMDEncodedAutoencoderPL(BallsAutoencoderPL):
    def __init__(
        self,
        top_k: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)

        self.penalty_weight = self.hparams.get("penalty_weight", 1.0)
        self.wait_steps = self.hparams.get("wait_steps", 0)
        self.linear_steps = self.hparams.get("linear_steps", 1)

        self.save_encoded_data = self.hparams.get("save_encoded_data", True)
        self.loaded_img_data = False
        self.validation_step_outputs = []

    def loss(self, x, x_hat, z_hat, domains):

        # x, x_hat: [batch_size, x_dim]
        reconstruction_loss = F.mse_loss(x_hat, x, reduction="mean")
        penalty_loss_value = torch.tensor(0., device=self.device)
        hinge_loss_value = hinge_loss(z_hat, domains, self.num_domains, self.z_dim_invariant_model, self.stddev_threshold, self.stddev_eps, self.hinge_loss_weight) if self.hinge_loss_weight > 0. else torch.tensor(0., device=x.device)

        if self.penalty_criterion and self.penalty_criterion["minmax"]:
            penalty_loss_args = [self.top_k, self.loss_transform]
            penalty_loss_value_ = penalty_loss_minmax(z_hat, domains, self.num_domains, self.z_dim_invariant_model, *penalty_loss_args)
            penalty_loss_value += penalty_loss_value_
            self.log(f"penalty_loss_minmax", penalty_loss_value_.item(), prog_bar=True)
        if self.penalty_criterion and self.penalty_criterion["stddev"]:
            penalty_loss_args = []
            penalty_loss_value_ = penalty_loss_stddev(z_hat, domains, self.num_domains, self.z_dim_invariant_model, *penalty_loss_args)
            penalty_loss_value += penalty_loss_value_
        if self.penalty_criterion and self.penalty_criterion["mmd"]:
            penalty_loss_args = []
            penalty_loss_value_ = mmd_loss(self.MMD, z_hat, domains, self.num_domains, self.z_dim_invariant_model, *penalty_loss_args)
            penalty_loss_value += penalty_loss_value_
            self.log(f"penalty_loss_mmd", penalty_loss_value_.item(), prog_bar=True)

        
        penalty_loss_value = penalty_loss_value * self.penalty_weight
        hinge_loss_value = hinge_loss_value * self.hinge_loss_weight
        loss = reconstruction_loss + penalty_loss_value + hinge_loss_value
        return loss, reconstruction_loss, penalty_loss_value, hinge_loss_value

    def on_training_start(self, *args, **kwargs):
        self.log(f"val_reconstruction_loss", 0.0)
        self.log(f"valid_penalty_loss", 0.0)
        self.log(f"val_loss", 0.0)
        return

    def training_step(self, train_batch, batch_idx):
        # x: [batch_size, z_dim]
        x, domains = train_batch["x"], train_batch["domain"]

        # z: [batch_size, latent_dim]
        # x_hat: [batch_size, z_dim]
        z_hat, x_hat = self(x)
        if batch_idx % 20 == 0:
            log.info(f"x.max(): {x.max()}, x_hat.max(): {x_hat.max()}, x.min(): {x.min()}, x_hat.min(): {x_hat.min()}, x.mean(): {x.mean()}, x_hat.mean(): {x_hat.mean()}")
            log.info(f"z_hat.max(): {z_hat.max()}, z_hat.min(): {z_hat.min()}, z_hat.mean(): {z_hat.mean()}, z_hat.std(): {z_hat.std()}")
            log.info(f"sample z_hat 1: {z_hat[50:55, 50:55]}")
            log.info(f"sample z_hat 2: {z_hat[50:55, :]}")
        loss, reconstruction_loss, penalty_loss_value, hinge_loss_value = self.loss(x, x_hat, z_hat, domains)
        self.log(f"train_reconstruction_loss", reconstruction_loss.item())
        self.log(f"train_penalty_loss", penalty_loss_value.item())
        self.log(f"train_hinge_loss", hinge_loss_value.item())
        self.log(f"train_loss", loss.item(), prog_bar=True)

        if self.save_encoded_data:
            # self.training_step_outputs.append({"z_hat":z_hat})
            self.training_step_outputs.append({"z_hat":z_hat, "z_invariant":train_batch["z_invariant"], "z_spurious":train_batch["z_spurious"], "z":train_batch["z"]})

        return loss

    def validation_step(self, valid_batch, batch_idx):
        # images: [batch_size, num_channels, width, height]
        x, z_invariant, z_spurious, domain, color = valid_batch["x"], valid_batch["z_invariant"], valid_batch["z_spurious"], valid_batch["domain"], valid_batch["color"]

        # z_hat: [batch_size, latent_dim]
        # x_hat: [batch_size, x_dim]
        z_hat, x_hat = self(x)

        if batch_idx % 20 == 0:
            if self.penalty_criterion and (self.penalty_criterion["minmax"] or self.penalty_criterion["mmd"]):
                # print all z_hat mins of all domains
                log.info(f"============== z_hat min all domains ==============\n{[z_hat[(domain == i).squeeze(), :self.z_dim_invariant_model].min().detach().cpu().numpy().item() for i in range(self.num_domains)]}\n")
                # print all z_hat maxs of all domains
                log.info(f"============== z_hat max all domains ==============\n{[z_hat[(domain == i).squeeze(), :self.z_dim_invariant_model].max().detach().cpu().numpy().item() for i in range(self.num_domains)]}\n")
                log.info(f"============== ============== ============== ==============\n")
            if self.penalty_criterion and self.penalty_criterion["stddev"] == 1.:
                # print all z_hat stds of all domains for each of z_dim_invariant dimensions
                for dim in range(self.z_dim_invariant_model):
                    log.info(f"============== z_hat std all domains dim {dim} ==============\n{[z_hat[(domain == i).squeeze(), dim].std().detach().cpu().numpy().item() for i in range(self.num_domains)]}\n")
                log.info(f"============== ============== ============== ==============\n")
            if self.penalty_criterion and self.penalty_criterion["domain_classification"] == 1.:
                # log the weigth matrix of the multinomial logistic regression model
                log.info(f"============== multinomial logistic regression model weight matrix ==============\n{self.multinomial_logistic_regression.linear.weight}\n")

        # if batch_idx % 5 == 0:
        #     # fit a linear regression from z_hat on z
        #     z = valid_batch["z"] # [batch_size, n_balls * z_dim_ball]
        #     r2, mse_loss = self.compute_r2(z, z_hat)
        #     self.log(f"r2_linreg", r2, prog_bar=False)
        #     self.log(f"mse_loss_linreg", mse_loss, prog_bar=False)
        #     r2, mse_loss = self.compute_r2(z_hat, z)
        #     self.log(f"~r2_linreg", r2, prog_bar=True)
        #     self.log(f"~mse_loss_linreg", mse_loss, prog_bar=False)

        #     # fit a linear regression from z_hat invariant dims to z_invariant dimensions
        #     # z_invariant: [batch_size, n_balls_invariant * z_dim_ball]
        #     r2, mse_loss = self.compute_r2(z_invariant, z_hat[:, :self.z_dim_invariant_model])
        #     self.log(f"hz_z_r2_linreg", r2, prog_bar=False)
        #     self.log(f"hz_z_mse_loss_linreg", mse_loss, prog_bar=False)
        #     r2, mse_loss = self.compute_r2(z_hat[:, :self.z_dim_invariant_model], z_invariant)
        #     self.log(f"hz_z_~r2_linreg", r2, prog_bar=True)
        #     self.log(f"hz_z_~mse_loss_linreg", mse_loss, prog_bar=False)

        #     # fit a linear regression from z_hat invariant dims to z_spurious dimensions
        #     # z_spurious: [batch_size, n_balls_spurious * z_dim_ball]
        #     r2, mse_loss = self.compute_r2(z_spurious, z_hat[:, :self.z_dim_invariant_model])
        #     self.log(f"hz_~z_r2_linreg", r2, prog_bar=False)
        #     self.log(f"hz_~z_mse_loss_linreg", mse_loss, prog_bar=False)
        #     r2, mse_loss = self.compute_r2(z_hat[:, :self.z_dim_invariant_model], z_spurious)
        #     self.log(f"hz_~z_~r2_linreg", r2, prog_bar=True)
        #     self.log(f"hz_~z_~mse_loss_linreg", mse_loss, prog_bar=False)

        #     # fit a linear regression from z_hat spurious dims to z_invariant dimensions
        #     # z_invariant: [batch_size, n_balls_invariant * z_dim_ball]
        #     r2, mse_loss = self.compute_r2(z_invariant, z_hat[:, self.z_dim_invariant_model:])
        #     self.log(f"~hz_z_r2_linreg", r2, prog_bar=False)
        #     self.log(f"~hz_z_mse_loss_linreg", mse_loss, prog_bar=False)
        #     r2, mse_loss = self.compute_r2(z_hat[:, self.z_dim_invariant_model:], z_invariant)
        #     self.log(f"~hz_z_~r2_linreg", r2, prog_bar=False)
        #     self.log(f"~hz_z_~mse_loss_linreg", mse_loss, prog_bar=False)
            
        #     # fit a linear regression from z_hat spurious dims to z_spurious dimensions
        #     # z_spurious: [batch_size, n_balls_spurious * z_dim_ball]
        #     r2, mse_loss = self.compute_r2(z_spurious, z_hat[:, self.z_dim_invariant_model:])
        #     self.log(f"~hz_~z_r2_linreg", r2, prog_bar=False)
        #     self.log(f"~hz_~z_mse_loss_linreg", mse_loss, prog_bar=False)
        #     r2, mse_loss = self.compute_r2(z_hat[:, self.z_dim_invariant_model:], z_spurious)
        #     self.log(f"~hz_~z_~r2_linreg", r2, prog_bar=False)
        #     self.log(f"~hz_~z_~mse_loss_linreg", mse_loss, prog_bar=False)

        #     # comptue the average norm of first z_dim dimensions of z
        #     z_norm = torch.norm(z_hat[:, :self.z_dim_invariant_model], dim=1).mean()
        #     self.log(f"z_norm", z_norm, prog_bar=False)
        #     # comptue the average norm of the last n-z_dim dimensions of z
        #     z_norm = torch.norm(z_hat[:, self.z_dim_invariant_model:], dim=1).mean()
        #     self.log(f"~z_norm", z_norm, prog_bar=False)

        #     # compute the regression scores with MLPRegressor
        #     hidden_layer_size = 400 # z_hat.shape[1]

        #     # fit a MLP regression from z_hat on z
        #     # r2, reg_loss = self.compute_r2_mlp(z, z_hat, hidden_layer_size)
        #     # self.log(f"r2_mlpreg", r2, prog_bar=True)
        #     # self.log(f"reg_loss_mlpreg", reg_loss, prog_bar=True)
        #     # r2, reg_loss = self.compute_r2_mlp(z_hat, z, hidden_layer_size)
        #     # self.log(f"~r2_mlpreg", r2, prog_bar=True)
        #     # self.log(f"~reg_loss_mlpreg", reg_loss, prog_bar=True)

        #     # fit a MLP regression from z_hat invariant dims to z_invariant dimensions
        #     # r2, reg_loss = self.compute_r2_mlp(z_invariant, z_hat[:, :self.z_dim_invariant_model], hidden_layer_size)
        #     # self.log(f"hz_z_r2_mlpreg", r2, prog_bar=True)
        #     # self.log(f"hz_z_reg_loss_mlpreg", reg_loss, prog_bar=True)
        #     # r2, reg_loss = self.compute_r2_mlp(z_hat[:, :self.z_dim_invariant_model], z_invariant, hidden_layer_size)
        #     # self.log(f"hz_z_~r2_mlpreg", r2, prog_bar=True)
        #     # self.log(f"hz_z_~reg_loss_mlpreg", reg_loss, prog_bar=True)

        #     # # fit a MLP regression from z_hat invariant dims to z_spurious dimensions
        #     # r2, reg_loss = self.compute_r2_mlp(z_spurious, z_hat[:, :self.z_dim_invariant_model], hidden_layer_size)
        #     # self.log(f"hz_~z_r2_mlpreg", r2, prog_bar=True)
        #     # self.log(f"hz_~z_reg_loss_mlpreg", reg_loss, prog_bar=True)
        #     # r2, reg_loss = self.compute_r2_mlp(z_hat[:, :self.z_dim_invariant_model], z_spurious, hidden_layer_size)
        #     # self.log(f"hz_~z_~reg_loss_mlpreg", r2, prog_bar=True)
        #     # self.log(f"hz_~z_~r2_mlpreg", r2, prog_bar=True)

        #     # # fit a MLP regression from z_hat spurious dims to z_invariant dimensions
        #     # r2, reg_loss = self.compute_r2_mlp(z_invariant, z_hat[:, self.z_dim_invariant_model:], hidden_layer_size)
        #     # self.log(f"~hz_z_r2_mlpreg", r2, prog_bar=True)
        #     # self.log(f"~hz_z_reg_loss_mlpreg", reg_loss, prog_bar=True)
        #     # r2, reg_loss = self.compute_r2_mlp(z_hat[:, self.z_dim_invariant_model:], z_invariant, hidden_layer_size)
        #     # self.log(f"~hz_z_~reg_loss_mlpreg", r2, prog_bar=True)
        #     # self.log(f"~hz_z_~r2_mlpreg", r2, prog_bar=True)

        #     # # fit a MLP regression from z_hat spurious dims to z_spurious dimensions
        #     # r2, reg_loss = self.compute_r2_mlp(z_spurious, z_hat[:, self.z_dim_invariant_model:], hidden_layer_size)
        #     # self.log(f"~hz_~z_r2_mlpreg", r2, prog_bar=True)
        #     # self.log(f"~hz_~z_reg_loss_mlpreg", reg_loss, prog_bar=True)
        #     # r2, reg_loss = self.compute_r2_mlp(z_hat[:, self.z_dim_invariant_model:], z_spurious, hidden_layer_size)
        #     # self.log(f"~hz_~z_~reg_loss_mlpreg", r2, prog_bar=True)
        #     # self.log(f"~hz_~z_~r2_mlpreg", r2, prog_bar=True)

        #     # compute domain classification accuracy with multinoimal logistic regression for z_hat, z_invariant, z_spurious
        #     acc = self.compute_acc_logistic_regression(z_hat, domain)
        #     self.log(f"domain_acc_logreg", acc, prog_bar=False)

        #     # domain prediction from zhat z_dim_invariant dimensions
        #     acc = self.compute_acc_logistic_regression(z_hat[:, :self.z_dim_invariant_model], domain)
        #     self.log(f"hz_domain_acc_logreg", acc, prog_bar=True)

        #     # domain prediction from zhat z_dim_spurious dimensions
        #     acc = self.compute_acc_logistic_regression(z_hat[:, self.z_dim_invariant_model:], domain)
        #     self.log(f"~hz_domain_acc_logreg", acc, prog_bar=True)

        #     # compute domain classification accuracy with MLPClassifier for z_hat, z_invariant, z_spurious
        #     # acc = self.compute_acc_mlp(z_hat, domain, hidden_layer_size)
        #     # self.log(f"domain_acc_mlp", acc, prog_bar=False)

        #     # # domain prediction from zhat z_dim_invariant dimensions
        #     # acc = self.compute_acc_mlp(z_hat[:, :self.z_dim_invariant_model], domain, hidden_layer_size)
        #     # self.log(f"hz_domain_acc_mlp", acc, prog_bar=False)

        #     # # domain prediction from zhat z_dim_spurious dimensions
        #     # acc = self.compute_acc_mlp(z_hat[:, self.z_dim_invariant_model:], domain, hidden_layer_size)
        #     # self.log(f"~hz_domain_acc_mlp", acc, prog_bar=False)


        loss, reconstruction_loss, penalty_loss_value, hinge_loss_value = self.loss(x, x_hat, z_hat, domain)
        self.log(f"val_reconstruction_loss", reconstruction_loss.item())
        self.log(f"val_penalty_loss", penalty_loss_value.item())
        self.log(f"val_hinge_loss", hinge_loss_value.item())
        self.log(f"val_loss", loss.item())

        if self.save_encoded_data:
            self.validation_step_outputs.append({"z_hat":z_hat, "z_invariant":z_invariant, "z_spurious":z_spurious, "z":valid_batch["z"]})
            # self.validation_step_outputs.append({"z_hat":z_hat})
        else:
            self.validation_step_outputs.append({"z_hat":z_hat, "z_invariant":z_invariant, "z_spurious":z_spurious, "z":valid_batch["z"]})

        return loss


    def on_train_epoch_end(self):

        if self.save_encoded_data:
            # load the train dataset, and replace its "image" key with the new_data["z_hat"] key
            # and save it as a pt file

            new_data = dict.fromkeys(self.train_dataset.keys())

            # stack the values of each key in the new_data dict
            # new_data is a list of dicts
            for key in new_data.keys():
                new_data[key] = self.train_dataset[key]
            for key in self.training_step_outputs[0].keys():
                new_data.pop(key, None)
            for key in self.training_step_outputs[0].keys():
                new_data[key] = torch.zeros((len(self.train_dataset[key]), self.training_step_outputs[0][key].shape[-1]))

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
            torch.save(new_data, os.path.join(os.getcwd(), f"double_encoded_img_{self.trainer.datamodule.datamodule_name}_train_epoch_{self.trainer.current_epoch}.pt"))
        self.training_step_outputs.clear()

        return
    
    def on_validation_epoch_end(self):

        if self.save_encoded_data:
            # load the valid dataset, and replace its "image" key with the new_data["z_hat"] key
            # and save it as a pt file
            if not self.loaded_img_data:
                try:
                    path = os.path.join(self.trainer.datamodule.path_to_files, f"encoded_img_md_balls_valid.pt")
                    self.valid_dataset = torch.load(path).dataset
                    log.info(f"Loaded the validation dataset of length {len(self.valid_dataset)} from: {path}")
                    path = os.path.join(self.trainer.datamodule.path_to_files, f"encoded_img_md_balls_train.pt")
                    self.train_dataset = torch.load(path).dataset
                    log.info(f"Loaded the training dataset of length {len(self.train_dataset)} from: {path}")
                except AttributeError:
                    path = os.path.join(self.trainer.datamodule.path_to_files, f"encoded_img_md_balls_valid.pt")
                    self.valid_dataset = torch.load(path)
                    log.info(f"Loaded the training dataset of length {len(self.valid_dataset)} from: {path}")
                    path = os.path.join(self.trainer.datamodule.path_to_files, f"encoded_img_md_balls_train.pt")
                    self.train_dataset = torch.load(path)
                    log.info(f"Loaded the training dataset of length {len(self.train_dataset)} from: {path}")
                self.loaded_img_data = True

            new_data = dict.fromkeys(self.valid_dataset.keys())

            # stack the values of each key in the new_data dict
            # new_data is a list of dicts
            for key in new_data.keys():
                new_data[key] = self.valid_dataset[key]
            for key in self.validation_step_outputs[0].keys():
                new_data.pop(key, None)
            for key in self.validation_step_outputs[0].keys():
                new_data[key] = torch.zeros((len(self.valid_dataset[key]), self.validation_step_outputs[0][key].shape[-1]))
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
            torch.save(new_data, os.path.join(os.getcwd(), f"double_encoded_img_{self.trainer.datamodule.datamodule_name}_valid_epoch_{self.trainer.current_epoch}.pt"))

        z = torch.cat([output["z"] for output in self.validation_step_outputs], dim=0)
        z_hat = torch.cat([output["z_hat"] for output in self.validation_step_outputs], dim=0)
        z_invariant = torch.cat([output["z_invariant"] for output in self.validation_step_outputs], dim=0)
        z_spurious = torch.cat([output["z_spurious"] for output in self.validation_step_outputs], dim=0)

        r2, mse_loss = self.compute_r2(z, z_hat)
        self.log(f"r2_linreg", r2, prog_bar=False)
        self.log(f"mse_loss_linreg", mse_loss, prog_bar=False)
        r2, mse_loss = self.compute_r2(z_hat, z)
        self.log(f"~r2_linreg", r2, prog_bar=True)
        self.log(f"~mse_loss_linreg", mse_loss, prog_bar=False)

        # fit a linear regression from z_hat invariant dims to z_invariant dimensions
        # z_invariant: [batch_size, n_balls_invariant * z_dim_ball]
        r2, mse_loss = self.compute_r2(z_invariant, z_hat[:, :self.z_dim_invariant_model])
        self.log(f"hz_z_r2_linreg", r2, prog_bar=False)
        self.log(f"hz_z_mse_loss_linreg", mse_loss, prog_bar=False)
        r2, mse_loss = self.compute_r2(z_hat[:, :self.z_dim_invariant_model], z_invariant)
        self.log(f"hz_z_~r2_linreg", r2, prog_bar=True)
        self.log(f"hz_z_~mse_loss_linreg", mse_loss, prog_bar=False)

        # fit a linear regression from z_hat invariant dims to z_spurious dimensions
        # z_spurious: [batch_size, n_balls_spurious * z_dim_ball]
        r2, mse_loss = self.compute_r2(z_spurious, z_hat[:, :self.z_dim_invariant_model])
        self.log(f"hz_~z_r2_linreg", r2, prog_bar=False)
        self.log(f"hz_~z_mse_loss_linreg", mse_loss, prog_bar=False)
        r2, mse_loss = self.compute_r2(z_hat[:, :self.z_dim_invariant_model], z_spurious)
        self.log(f"hz_~z_~r2_linreg", r2, prog_bar=True)
        self.log(f"hz_~z_~mse_loss_linreg", mse_loss, prog_bar=False)

        # # concatenate all z_hat, z, z_invariant, z_spurious, domain
        # z_hat = torch.cat([output["z_hat"] for output in self.validation_step_outputs], dim=0)
        # z = torch.cat([output["z"] for output in self.validation_step_outputs], dim=0)
        # z_invariant = torch.cat([output["z_invariant"] for output in self.validation_step_outputs], dim=0)
        # z_spurious = torch.cat([output["z_spurious"] for output in self.validation_step_outputs], dim=0)
        # domain = torch.cat([output["domain"] for output in self.validation_step_outputs], dim=0)

        # # fit a linear regression from z_hat on z
        # r2 = self.compute_r2(z, z_hat)
        # self.log(f"r2_linreg", r2, prog_bar=True)
        # r2 = self.compute_r2(z_hat, z)
        # self.log(f"~r2_linreg", r2, prog_bar=True)

        # # fit a linear regression from z_hat invariant dims to z_invariant dimensions
        # # z_invariant: [batch_size, n_balls_invariant * z_dim_ball]
        # r2 = self.compute_r2(z_invariant, z_hat[:, :self.z_dim_invariant_model])
        # self.log(f"hz_z_r2_linreg", r2, prog_bar=True)
        # r2 = self.compute_r2(z_hat[:, :self.z_dim_invariant_model], z_invariant)
        # self.log(f"hz_z_~r2_linreg", r2, prog_bar=True)

        # # fit a linear regression from z_hat invariant dims to z_spurious dimensions
        # # z_spurious: [batch_size, n_balls_spurious * z_dim_ball]
        # r2 = self.compute_r2(z_spurious, z_hat[:, :self.z_dim_invariant_model])
        # self.log(f"hz_~z_r2_linreg", r2, prog_bar=True)
        # r2 = self.compute_r2(z_hat[:, :self.z_dim_invariant_model], z_spurious)
        # self.log(f"hz_~z_~r2_linreg", r2, prog_bar=True)

        # # fit a linear regression from z_hat spurious dims to z_invariant dimensions
        # # z_invariant: [batch_size, n_balls_invariant * z_dim_ball]
        # r2 = self.compute_r2(z_invariant, z_hat[:, self.z_dim_invariant_model:])
        # self.log(f"~hz_z_r2_linreg", r2, prog_bar=True)
        # r2 = self.compute_r2(z_hat[:, self.z_dim_invariant_model:], z_invariant)
        # self.log(f"~hz_z_~r2_linreg", r2, prog_bar=True)
        
        # # fit a linear regression from z_hat spurious dims to z_spurious dimensions
        # # z_spurious: [batch_size, n_balls_spurious * z_dim_ball]
        # r2 = self.compute_r2(z_spurious, z_hat[:, self.z_dim_invariant_model:])
        # self.log(f"~hz_~z_r2_linreg", r2, prog_bar=True)
        # r2 = self.compute_r2(z_hat[:, self.z_dim_invariant_model:], z_spurious)
        # self.log(f"~hz_~z_~r2_linreg", r2, prog_bar=True)

        # # comptue the average norm of first z_dim dimensions of z
        # z_norm = torch.norm(z_hat[:, :self.z_dim_invariant_model], dim=1).mean()
        # self.log(f"z_norm", z_norm, prog_bar=False)
        # # comptue the average norm of the last n-z_dim dimensions of z
        # z_norm = torch.norm(z_hat[:, self.z_dim_invariant_model:], dim=1).mean()
        # self.log(f"~z_norm", z_norm, prog_bar=False)

        # # compute the regression scores with MLPRegressor
        # hidden_layer_size = 128 # z_hat.shape[1]

        # # fit a MLP regression from z_hat on z
        # r2, reg_loss = self.compute_r2_mlp(z, z_hat, hidden_layer_size)
        # self.log(f"r2_mlpreg", r2, prog_bar=True)
        # self.log(f"reg_loss_mlpreg", reg_loss, prog_bar=True)
        # r2, reg_loss = self.compute_r2_mlp(z_hat, z, hidden_layer_size)
        # self.log(f"~r2_mlpreg", r2, prog_bar=True)
        # self.log(f"~reg_loss_mlpreg", reg_loss, prog_bar=True)

        # # fit a MLP regression from z_hat invariant dims to z_invariant dimensions
        # r2, reg_loss = self.compute_r2_mlp(z_invariant, z_hat[:, :self.z_dim_invariant_model], hidden_layer_size)
        # self.log(f"hz_z_r2_mlpreg", r2, prog_bar=True)
        # self.log(f"hz_z_reg_loss_mlpreg", reg_loss, prog_bar=True)
        # r2, reg_loss = self.compute_r2_mlp(z_hat[:, :self.z_dim_invariant_model], z_invariant, hidden_layer_size)
        # self.log(f"hz_z_~r2_mlpreg", r2, prog_bar=True)
        # self.log(f"hz_z_~reg_loss_mlpreg", reg_loss, prog_bar=True)

        # # fit a MLP regression from z_hat invariant dims to z_spurious dimensions
        # r2, reg_loss = self.compute_r2_mlp(z_spurious, z_hat[:, :self.z_dim_invariant_model], hidden_layer_size)
        # self.log(f"hz_~z_r2_mlpreg", r2, prog_bar=True)
        # self.log(f"hz_~z_reg_loss_mlpreg", reg_loss, prog_bar=True)
        # r2, reg_loss = self.compute_r2_mlp(z_hat[:, :self.z_dim_invariant_model], z_spurious, hidden_layer_size)
        # self.log(f"hz_~z_~reg_loss_mlpreg", r2, prog_bar=True)
        # self.log(f"hz_~z_~r2_mlpreg", r2, prog_bar=True)

        # # fit a MLP regression from z_hat spurious dims to z_invariant dimensions
        # r2, reg_loss = self.compute_r2_mlp(z_invariant, z_hat[:, self.z_dim_invariant_model:], hidden_layer_size)
        # self.log(f"~hz_z_r2_mlpreg", r2, prog_bar=True)
        # self.log(f"~hz_z_reg_loss_mlpreg", reg_loss, prog_bar=True)
        # r2, reg_loss = self.compute_r2_mlp(z_hat[:, self.z_dim_invariant_model:], z_invariant, hidden_layer_size)
        # self.log(f"~hz_z_~reg_loss_mlpreg", r2, prog_bar=True)
        # self.log(f"~hz_z_~r2_mlpreg", r2, prog_bar=True)

        # # fit a MLP regression from z_hat spurious dims to z_spurious dimensions
        # r2, reg_loss = self.compute_r2_mlp(z_spurious, z_hat[:, self.z_dim_invariant_model:], hidden_layer_size)
        # self.log(f"~hz_~z_r2_mlpreg", r2, prog_bar=True)
        # self.log(f"~hz_~z_reg_loss_mlpreg", reg_loss, prog_bar=True)
        # r2, reg_loss = self.compute_r2_mlp(z_hat[:, self.z_dim_invariant_model:], z_spurious, hidden_layer_size)
        # self.log(f"~hz_~z_~reg_loss_mlpreg", r2, prog_bar=True)
        # self.log(f"~hz_~z_~r2_mlpreg", r2, prog_bar=True)

        # # compute domain classification accuracy with multinoimal logistic regression for z_hat, z_invariant, z_spurious
        # acc = self.compute_acc_logistic_regression(z_hat, domain)
        # self.log(f"domain_acc_logreg", acc, prog_bar=False)

        # # domain prediction from zhat z_dim_invariant dimensions
        # acc = self.compute_acc_logistic_regression(z_hat[:, :self.z_dim_invariant_model], domain)
        # self.log(f"hz_domain_acc_logreg", acc, prog_bar=True)

        # # domain prediction from zhat z_dim_spurious dimensions
        # acc = self.compute_acc_logistic_regression(z_hat[:, self.z_dim_invariant_model:], domain)
        # self.log(f"~hz_domain_acc_logreg", acc, prog_bar=True)

        # # compute domain classification accuracy with MLPClassifier for z_hat, z_invariant, z_spurious
        # acc = self.compute_acc_mlp(z_hat, domain, hidden_layer_size)
        # self.log(f"domain_acc_mlp", acc, prog_bar=False)

        # # domain prediction from zhat z_dim_invariant dimensions
        # acc = self.compute_acc_mlp(z_hat[:, :self.z_dim_invariant_model], domain, hidden_layer_size)
        # self.log(f"hz_domain_acc_mlp", acc, prog_bar=False)

        # # domain prediction from zhat z_dim_spurious dimensions
        # acc = self.compute_acc_mlp(z_hat[:, self.z_dim_invariant_model:], domain, hidden_layer_size)
        # self.log(f"~hz_domain_acc_mlp", acc, prog_bar=False)
        self.validation_step_outputs.clear()

        return

    def on_train_start(self):

        # log the r2 scores before any training has started
        valid_dataset = self.trainer.datamodule.valid_dataset

        z = valid_dataset.data["z"]
        z_hat = valid_dataset.data["z_hat"]
        z_invariant = valid_dataset.data["z_invariant"]
        z_spurious = valid_dataset.data["z_spurious"]

        r2, mse_loss = self.compute_r2(z, z_hat)
        self.log(f"r2_linreg_start", r2, prog_bar=False)
        # self.log(f"mse_loss_linreg_start", mse_loss, prog_bar=False)
        r2, mse_loss = self.compute_r2(z_hat, z)
        self.log(f"~r2_linreg_start", r2, prog_bar=False)
        # self.log(f"~mse_loss_linreg_start", mse_loss, prog_bar=False)

        # fit a linear regression from z_hat invariant dims to z_invariant dimensions
        # z_invariant: [batch_size, n_balls_invariant * z_dim_ball]
        r2, mse_loss = self.compute_r2(z_invariant, z_hat[:, :self.z_dim_invariant_model])
        self.log(f"hz_z_r2_linreg_start", r2, prog_bar=False)
        # self.log(f"hz_z_mse_loss_linreg_start", mse_loss, prog_bar=False)
        r2, mse_loss = self.compute_r2(z_hat[:, :self.z_dim_invariant_model], z_invariant)
        self.log(f"hz_z_~r2_linreg_start", r2, prog_bar=False)
        # self.log(f"hz_z_~mse_loss_linreg", mse_loss, prog_bar=False)

        # fit a linear regression from z_hat invariant dims to z_spurious dimensions
        # z_spurious: [batch_size, n_balls_spurious * z_dim_ball]
        r2, mse_loss = self.compute_r2(z_spurious, z_hat[:, :self.z_dim_invariant_model])
        self.log(f"hz_~z_r2_linreg_start", r2, prog_bar=False)
        # self.log(f"hz_~z_mse_loss_linreg_start", mse_loss, prog_bar=False)
        r2, mse_loss = self.compute_r2(z_hat[:, :self.z_dim_invariant_model], z_spurious)
        self.log(f"hz_~z_~r2_linreg_start", r2, prog_bar=False)
        # self.log(f"hz_~z_~mse_loss_linreg_start", mse_loss, prog_bar=False)

        return
