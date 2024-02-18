# pylint: disable-all
import torch
from torch.nn import functional as F
from .base_pl import BasePl
import hydra
from omegaconf import OmegaConf
import os
import wandb
from utils.disentanglement_utils import linear_disentanglement, permutation_disentanglement
import utils.general as utils
log = utils.get_logger(__name__)


class MixingAutoencoderPL(BasePl):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.model = hydra.utils.instantiate(self.hparams.autoencoder, _recursive_=False) # type: ignore
        if self.hparams.get("autoencoder_ckpt_path", None) is not None:    
            ckpt_path = self.hparams["autoencoder_ckpt_path"]
            # only load the weights, i.e. HPs should be overwritten from the passed config
            # b/c maybe the ckpt has num_slots=7, but we want to test it w/ num_slots=12
            # NOTE: NEVER DO self.model = self.model.load_state_dict(...), raises _IncompatibleKey error
            self.model.load_state_dict(torch.load(ckpt_path))
            self.hparams.pop("autoencoder_ckpt_path") # we don't want this to be save with the ckpt, sicne it will raise key errors when we further train the model
                                                  # and load it for evaluation.

        # freeze the parameters of encoder if needed
        if self.hparams.get("autoencoder_freeze", False):
            for param in self.model.parameters():
                param.requires_grad = False

        else: # if the flag is set to true we should correct the requires_grad flags, i.e. we might
              # initially freeze it for some time, but then decide to let it finetune.
            for param in self.model.parameters():
                param.requires_grad = True

        self.num_domains = self.hparams.get("num_domains", 4)
        self.z_dim_invariant_model = self.hparams.get("z_dim_invariant", 2)
        self.save_encoded_data = self.hparams.get("save_encoded_data", False)
        self.penalty_weight = self.hparams.get("penalty_weight", 1.0)
        self.wait_steps = int(self.hparams.get("wait_steps", 0))
        self.linear_steps = int(self.hparams.get("linear_steps", 1))

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):

        # x: [batch_size, x_dim]
        x, z, domain = train_batch["x"], train_batch["z"], train_batch["domain"]

        # if not self.domain_classification_flag:

        # z: [batch_size, latent_dim]
        # x_hat: [batch_size, x_dim]
        z_hat, x_hat = self(x)

        # adjusting self.penalty_weight according to the warmup schedule during training
        if self.trainer.global_step < self.wait_steps:
            self.penalty_weight = 0.0
        elif self.wait_steps <= self.trainer.global_step < self.wait_steps + self.linear_steps:
            self.penalty_weight = self.hparams.penalty_weight * (self.trainer.global_step - self.wait_steps) / self.linear_steps
        else:
            self.penalty_weight = self.hparams.penalty_weight
        self.log("penalty_weight", self.penalty_weight)

        loss, reconstruction_loss, penalty_loss_value, hinge_loss_value = self.loss(x, x_hat, z_hat, domain)
        self.log(f"train_reconstruction_loss", reconstruction_loss.item())
        self.log(f"train_penalty_loss", penalty_loss_value.item())
        self.log(f"train_hinge_loss", hinge_loss_value.item())
        self.log(f"train_loss", loss.item(), prog_bar=True)

        if self.save_encoded_data:
            self.training_step_outputs.append({"z_hat":z_hat})

        return loss
        # else:
        #     self.manual_adversarial_training_step(x, domain, batch_idx)

    def validation_step(self, valid_batch, batch_idx):

        # x: [batch_size, x_dim]
        x, z, domain = valid_batch["x"], valid_batch["z"], valid_batch["domain"]


        # z: [batch_size, latent_dim]
        # x_hat: [batch_size, x_dim]
        z_hat, x_hat = self(x)

        if batch_idx % 20 == 0:
            if self.penalty_criterion and (self.penalty_criterion["minmax"] or self.penalty_criterion["mmd"]):
                # your code here
                # print all z_hat mins of all domains
                print(f"============== z_hat min all domains ==============\n{[z_hat[(domain == i).squeeze(), :self.z_dim_invariant_model].min().detach().cpu().numpy().item() for i in range(self.num_domains)]}\n")
                # print all z_hat maxs of all domains
                print(f"============== z_hat max all domains ==============\n{[z_hat[(domain == i).squeeze(), :self.z_dim_invariant_model].max().detach().cpu().numpy().item() for i in range(self.num_domains)]}\n")
                print(f"============== ============== ============== ==============\n")
            if self.penalty_criterion and self.penalty_criterion["stddev"]:
                # your code here
                # print all z_hat stds of all domains for each of z_dim_invariant dimensions
                for dim in range(self.z_dim_invariant_model): # type: ignore
                    print(f"============== z_hat std all domains dim {dim} ==============\n{[z_hat[(domain == i).squeeze(), dim].std().detach().cpu().numpy().item() for i in range(self.num_domains)]}\n")
                print(f"============== ============== ============== ==============\n")
            if self.penalty_criterion and self.penalty_criterion["domain_classification"]:
                # print the accuracy of classifying domains from z_hat[:, :z_dim_invariant]
                from sklearn.linear_model import LogisticRegression
                # import accuracy from metrics
                from sklearn.metrics import accuracy_score
                clf = LogisticRegression(random_state=0, max_iter=1000).fit(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), domain.detach().cpu().numpy())
                pred_domain = clf.predict(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy())
                accuracy = accuracy_score(domain.detach().cpu().numpy(), pred_domain)
                print(f"============== z_hat domain classification accuracy: {accuracy} ==============")


        # we have the set of z and z_hat. We want to train a linear regression to predict the
        # z from z_hat using sklearn, and report regression scores. We do the same with mlpReg
        # import linear regression from sklearn
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import accuracy_score, r2_score
        self.z_dim_invariant_data = self.trainer.datamodule.train_dataset.z_dim_invariant # type: ignore

        reg = LinearRegression().fit(z.detach().cpu().numpy(), z_hat.detach().cpu().numpy())
        r2 = reg.score(z.detach().cpu().numpy(), z_hat.detach().cpu().numpy())
        self.log(f"r2", r2, prog_bar=True)
        reg = LinearRegression().fit(z_hat.detach().cpu().numpy(), z.detach().cpu().numpy())
        r2 = reg.score(z_hat.detach().cpu().numpy(), z.detach().cpu().numpy())
        self.log(f"~r2", r2, prog_bar=True)

        # we have 4 linear regression tasks:
        # 1. predicting z_hat[:z_dim_invariant] from z[:z_dim_invariant]
        reg = LinearRegression().fit(z[:, :self.z_dim_invariant_data].detach().cpu().numpy(), z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy())
        r2 = reg.score(z[:, :self.z_dim_invariant_data].detach().cpu().numpy(), z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy())
        self.log(f"hz_z_r2", r2, prog_bar=False)
        reg = LinearRegression().fit(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), z[:, :self.z_dim_invariant_data].detach().cpu().numpy())
        r2 = reg.score(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), z[:, :self.z_dim_invariant_data].detach().cpu().numpy())
        self.log(f"hz_z_~r2", r2, prog_bar=True)

        # 2. predicting z_hat[z_dim_invariant:] from z[:z_dim_invariant]
        reg = LinearRegression().fit(z[:, :self.z_dim_invariant_data].detach().cpu().numpy(), z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy())
        r2 = reg.score(z[:, :self.z_dim_invariant_data].detach().cpu().numpy(), z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy())
        self.log(f"~hz_z_r2", r2, prog_bar=False)
        reg = LinearRegression().fit(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), z[:, :self.z_dim_invariant_data].detach().cpu().numpy())
        r2 = reg.score(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), z[:, :self.z_dim_invariant_data].detach().cpu().numpy())
        self.log(f"~hz_z_~r2", r2, prog_bar=False)

        # 3. predicting z_hat[:z_dim_invariant] from z[z_dim_invariant:]
        reg = LinearRegression().fit(z[:, self.z_dim_invariant_data:].detach().cpu().numpy(), z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy())
        r2 = reg.score(z[:, self.z_dim_invariant_data:].detach().cpu().numpy(), z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy())
        self.log(f"hz_~z_r2", r2, prog_bar=False)
        reg = LinearRegression().fit(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), z[:, self.z_dim_invariant_data:].detach().cpu().numpy())
        r2 = reg.score(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), z[:, self.z_dim_invariant_data:].detach().cpu().numpy())
        self.log(f"hz_~z_~r2", r2, prog_bar=True)

        # 4. predicting z_hat[z_dim_invariant:] from z[z_dim_invariant:]
        reg = LinearRegression().fit(z[:, self.z_dim_invariant_data:].detach().cpu().numpy(), z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy())
        r2 = reg.score(z[:, self.z_dim_invariant_data:].detach().cpu().numpy(), z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy())
        self.log(f"~hz_~z_r2", r2, prog_bar=False)
        reg = LinearRegression().fit(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), z[:, self.z_dim_invariant_data:].detach().cpu().numpy())
        r2 = reg.score(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), z[:, self.z_dim_invariant_data:].detach().cpu().numpy())
        self.log(f"~hz_~z_~r2", r2, prog_bar=False)

        # # compute all of the above regression scores with MLPRegressor
        # from sklearn.neural_network import MLPRegressor
        # reg = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=z_hat.shape[1], activation='tanh').fit(z_hat.detach().cpu().numpy(), z.detach().cpu().numpy())
        # r2_score = reg.score(z_hat.detach().cpu().numpy(), z.detach().cpu().numpy())
        # self.log(f"val_r2_mlpreg", r2_score, prog_bar=True)

        # # we have 4 linear regression tasks:
        # # 1. predicting z[:z_dim_invariant] from z_hat[:z_dim_invariant]
        # reg = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=z_hat.shape[1], activation='tanh').fit(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), z[:, :self.z_dim_invariant_data].detach().cpu().numpy())
        # r2_score = reg.score(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), z[:, :self.z_dim_invariant_data].detach().cpu().numpy())
        # self.log(f"val_hz_z_r2_mlpreg", r2_score, prog_bar=True)

        # # 2. predicting z[:z_dim_invariant] from z_hat[z_dim_invariant:]
        # reg = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=z_hat.shape[1], activation='tanh').fit(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), z[:, :self.z_dim_invariant_data].detach().cpu().numpy())
        # r2_score = reg.score(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), z[:, :self.z_dim_invariant_data].detach().cpu().numpy())
        # self.log(f"val_~hz_z_r2_mlpreg", r2_score, prog_bar=True)

        # # 3. predicting z[:z_dim_invariant] from z_hat[:z_dim_invariant]
        # reg = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=z_hat.shape[1], activation='tanh').fit(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), z[:, self.z_dim_invariant_data:].detach().cpu().numpy())
        # r2_score = reg.score(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), z[:, self.z_dim_invariant_data:].detach().cpu().numpy())
        # self.log(f"val_hz_~z_r2_mlpreg", r2_score, prog_bar=True)

        # # 4. predicting z[z_dim_invariant:] from z_hat[z_dim_invariant:]
        # reg = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=z_hat.shape[1], activation='tanh').fit(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), z[:, self.z_dim_invariant_data:].detach().cpu().numpy())
        # r2_score = reg.score(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), z[:, self.z_dim_invariant_data:].detach().cpu().numpy())
        # self.log(f"val_~hz_~z_r2_mlpreg", r2_score, prog_bar=True)

        # comptue the average norm of first z_dim dimensions of z
        z_norm = torch.norm(z_hat[:, :self.z_dim_invariant_model], dim=1).mean()
        self.log(f"z_norm", z_norm, prog_bar=False)
        # comptue the average norm of the last n-z_dim dimensions of z
        z_norm = torch.norm(z_hat[:, self.z_dim_invariant_model:], dim=1).mean()
        self.log(f"~z_norm", z_norm, prog_bar=False)

        # compute and log the ratio of the variance of z along the z_dim_invariant_model dimensions to the expectatio
        # of norm of x
        z_var = torch.var(z_hat[:, :self.z_dim_invariant_model], dim=1).mean()
        x_norm = torch.norm(x, dim=1).mean()
        self.log(f"z_var/x_norm", z_var/x_norm, prog_bar=False)

        # do the same with the rest of z dimensions
        z_var = torch.var(z_hat[:, self.z_dim_invariant_model:], dim=1).mean()
        self.log(f"~z_var/x_norm", z_var/x_norm, prog_bar=False)

        loss, reconstruction_loss, penalty_loss_value, hinge_loss_value = self.loss(x, x_hat, z_hat, domain)
        self.log(f"val_reconstruction_loss", reconstruction_loss.item(), prog_bar=True)
        self.log(f"val_penalty_loss", penalty_loss_value.item(), prog_bar=True)
        try:
            self.log(f"val_hinge_loss", hinge_loss_value.item())
        except:
            self.log(f"val_hinge_loss", 0.0)
        self.log(f"val_loss", loss.item(), prog_bar=True)

        if self.save_encoded_data:
            self.validation_step_outputs.append({"z_hat":z_hat})

        return {"loss": loss}
    
    def on_train_epoch_end(self):

        if self.save_encoded_data:
            # load the train dataset, and replace its "x" key with the new_data["z_hat"] key
            # and save it as a pt file
            import os
            # train_dataset = torch.load(os.path.join(self.trainer.datamodule.path_to_files, f"train_dataset_{self.trainer.datamodule.datamodule_name}_{self.trainer.datamodule.num_samples['train']-self.trainer.datamodule.num_samples['valid']}.pt"))
            train_dataset = torch.load(os.path.join(os.getcwd(), f"train_dataset_{self.trainer.datamodule.datamodule_name}_{self.trainer.datamodule.num_samples['train']-self.trainer.datamodule.num_samples['valid']}.pt"))
            new_data = dict.fromkeys(train_dataset.data[0].keys())

            # stack the values of each key in the new_data dict
            # new_data is a list of dicts
            for key in new_data.keys():
                new_data[key] = torch.stack([torch.tensor(train_dataset.data[i][key]) for i in range(len(train_dataset))], dim=0)
            new_data.pop("x", None)
            for key in self.training_step_outputs[0].keys():
                new_data[key] = torch.zeros((len(train_dataset), self.training_step_outputs[0][key].shape[-1]))

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
            torch.save(new_data, os.path.join(os.getcwd(), f"encoded_{self.trainer.datamodule.datamodule_name}_train.pt"))
            self.training_step_outputs.clear()

        return
    
    def on_validation_epoch_end(self):

        if self.save_encoded_data:
            # load the valid dataset, and replace its "image" key with the new_data["z_hat"] key
            # and save it as a pt file
            import os
            try:
                # path = os.path.join(self.trainer.datamodule.path_to_files, f"valid_dataset_{self.trainer.datamodule.datamodule_name}_{self.trainer.datamodule.num_samples['valid']}.pt")
                path = os.path.join(os.getcwd(), f"valid_dataset_{self.trainer.datamodule.datamodule_name}_{self.trainer.datamodule.num_samples['valid']}.pt")
                valid_dataset = torch.load(path).dataset
                log.info(f"Loaded the validation dataset of length {len(valid_dataset)} from: {path}")
            except:
                # path = os.path.join(self.trainer.datamodule.path_to_files, f"valid_dataset_{self.trainer.datamodule.datamodule_name}_{self.trainer.datamodule.num_samples['valid']}.pt")
                path = os.path.join(os.getcwd(), f"valid_dataset_{self.trainer.datamodule.datamodule_name}_{self.trainer.datamodule.num_samples['valid']}.pt")
                valid_dataset = torch.load(path)
                log.info(f"Loaded the validation dataset of length {len(valid_dataset)} from: {path}")
            new_data = dict.fromkeys(valid_dataset.data[0].keys())

            # stack the values of each key in the new_data dict
            # new_data is a list of dicts
            for key in new_data.keys():
                new_data[key] = torch.stack([torch.tensor(valid_dataset.data[i][key]) for i in range(len(valid_dataset))], dim=0)
            new_data.pop("x", None)
            for key in self.validation_step_outputs[0].keys():
                new_data[key] = torch.zeros((len(valid_dataset), self.validation_step_outputs[0][key].shape[-1]))

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
            torch.save(new_data, os.path.join(os.getcwd(), f"encoded_{self.trainer.datamodule.datamodule_name}_valid.pt"))
            self.validation_step_outputs.clear()

        return

    def on_train_start(self):

        # log the r2 scores before any training has started
        valid_dataset = self.trainer.datamodule.valid_dataset
        # import code
        # code.interact(local=locals())
        z = torch.stack([t["z"] for t in valid_dataset.data], dim=0)
        x = torch.stack([t["x"] for t in valid_dataset.data], dim=0)
        z_hat, x_hat = self(x)

        z_invariant = z[:, :self.z_dim_invariant_model]
        z_spurious = z[:, self.z_dim_invariant_model:]

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
