import pytorch_lightning as pl
from models.utils import update
import torch
import hydra
from omegaconf import OmegaConf
from torch.nn import functional as F
import utils.general as utils
log = utils.get_logger(__name__)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, r2_score


class XZPl(pl.LightningModule):
    """
    A LightningModule organizes your PyTorch code into 5 sections:
        - Setup for all computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters(ignore=["datamodule", "additional_logger"], logger=False)
        # create a model with a sequential of n_layer linear layers followed by their respective activation
        # functions
        self.zhat_dim = self.hparams.get("zhat_dim", 128)
        self.zhat_dim_inv = self.hparams.get("zhat_dim_inv", 102)
        self.z_dim = self.hparams.get("z_dim", 4)
        self.z_dim_inv = self.hparams.get("z_dim_inv", 2)

        hidden_size = self.hparams.get("hidden_size", 500)
        

        n_layers = self.hparams.get("n_layers", 3)

        # z_hat to z
        input_layer = torch.nn.Linear(self.zhat_dim, hidden_size)
        output_layer = torch.nn.Linear(hidden_size, self.z_dim)
        linear_layers = [torch.nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)]
        activation_functions = [hydra.utils.instantiate(self.hparams.activation) for _ in range(n_layers)]
        # interleave the linear layers and activation functions
        hidden_layers = [layer for pair in zip(linear_layers, activation_functions) for layer in pair]
        self.model_zhat_z = torch.nn.Sequential(input_layer, *hidden_layers, output_layer)

        # z_hat_inv to z_inv
        input_layer = torch.nn.Linear(self.zhat_dim_inv, hidden_size)
        output_layer = torch.nn.Linear(hidden_size, self.z_dim_inv)
        linear_layers = [torch.nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)]
        activation_functions = [hydra.utils.instantiate(self.hparams.activation) for _ in range(n_layers)]
        # interleave the linear layers and activation functions
        hidden_layers = [layer for pair in zip(linear_layers, activation_functions) for layer in pair]
        self.model_zhat_inv_z_inv = torch.nn.Sequential(input_layer, *hidden_layers, output_layer)

        # z_hat_inv to z_spu
        input_layer = torch.nn.Linear(self.zhat_dim_inv, hidden_size)
        output_layer = torch.nn.Linear(hidden_size, self.z_dim_inv)
        linear_layers = [torch.nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)]
        activation_functions = [hydra.utils.instantiate(self.hparams.activation) for _ in range(n_layers)]
        # interleave the linear layers and activation functions
        hidden_layers = [layer for pair in zip(linear_layers, activation_functions) for layer in pair]
        self.model_zhat_inv_z_spu = torch.nn.Sequential(input_layer, *hidden_layers, output_layer)

        # turn off automatic optimization
        self.automatic_optimization = False

    def loss(self, z_hat, z):

        reconstruction_loss = F.mse_loss(z_hat, z, reduction="mean")
        return reconstruction_loss

    def forward(self, x, x_inv):

        return self.model_zhat_z(x), self.model_zhat_inv_z_inv(x_inv), self.model_zhat_inv_z_spu(x_inv)

    def validation_step(self, batch, batch_idx):

        zhat, z = batch["z_hat"], batch["z"]
        z_hat, z_hat_inv, z_hat_spu = self(zhat, zhat[:, :self.zhat_dim_inv])


        zhat_z_loss = self.loss(z_hat, z)
        # compute the r2 score of the prediction
        r2 = r2_score(z.detach().cpu().numpy(), z_hat.detach().cpu().numpy()) # y_true, y_pred
        self.log(f"val_r2", r2, prog_bar=True)
        self.log(f"val_zhat_z_loss", zhat_z_loss.item(), prog_bar=True)
        self.log(f"val_loss", zhat_z_loss.item(), prog_bar=True)

        z_hat_inv_z_inv_loss = self.loss(z_hat_inv, z[:, :self.z_dim_inv])
        r2 = r2_score(z[:, :self.z_dim_inv].detach().cpu().numpy(), z_hat_inv.detach().cpu().numpy())
        self.log(f"val_r2_inv", r2, prog_bar=True)
        self.log(f"val_zhat_inv_z_inv_loss", z_hat_inv_z_inv_loss.item(), prog_bar=True)
        
        z_hat_inv_z_spu_loss = self.loss(z_hat_spu, z[:, self.z_dim_inv:])
        r2 = r2_score(z[:, self.z_dim_inv:].detach().cpu().numpy(), z_hat_spu.detach().cpu().numpy())
        self.log(f"val_r2_spu", r2, prog_bar=True)
        self.log(f"val_zhat_inv_z_spu_loss", z_hat_inv_z_spu_loss.item(), prog_bar=True)

        return

    def configure_optimizers(self):
   
        model_zhat_z_params = []
        for param in self.model_zhat_z.parameters():
            if param.requires_grad:
                model_zhat_z_params.append(param)
        
        model_zhat_inv_z_inv_params = []
        for param in self.model_zhat_inv_z_inv.parameters():
            if param.requires_grad:
                model_zhat_inv_z_inv_params.append(param)
        
        model_zhat_inv_z_spu_params = []
        for param in self.model_zhat_inv_z_spu.parameters():
            if param.requires_grad:
                model_zhat_inv_z_spu_params.append(param)
        
        

        optimizer_zhat_z: torch.optim.Optimizer = hydra.utils.instantiate(self.hparams.optimizer
                                                                , model_zhat_z_params
                                                                )
        optimizer_zhat_inv_z_inv: torch.optim.Optimizer = hydra.utils.instantiate(self.hparams.optimizer
                                                                , model_zhat_inv_z_inv_params
                                                                )
        optimizer_zhat_inv_z_spu: torch.optim.Optimizer = hydra.utils.instantiate(self.hparams.optimizer
                                                                , model_zhat_inv_z_spu_params
                                                                )

        if self.hparams.get("scheduler_config"):
            if self.hparams.scheduler_config.scheduler['_target_'].startswith("torch.optim"):
                scheduler_zhat_z = hydra.utils.instantiate(self.hparams.scheduler_config.scheduler, optimizer_zhat_z)
                scheduler_zhat_inv_z_inv = hydra.utils.instantiate(self.hparams.scheduler_config.scheduler, optimizer_zhat_inv_z_inv)
                scheduler_zhat_inv_z_spu = hydra.utils.instantiate(self.hparams.scheduler_config.scheduler, optimizer_zhat_inv_z_spu)

            elif self.hparams.scheduler_config.scheduler['_target_'].startswith("transformers"):
                scheduler_zhat_z = hydra.utils.call(self.hparams.scheduler_config.scheduler, optimizer_zhat_z)
                scheduler_zhat_inv_z_inv = hydra.utils.call(self.hparams.scheduler_config.scheduler, optimizer_zhat_inv_z_inv)
                scheduler_zhat_inv_z_spu = hydra.utils.call(self.hparams.scheduler_config.scheduler, optimizer_zhat_inv_z_spu)
            
            else:
                raise ValueError("The scheduler specified by scheduler._target_ is not implemented.")
                
            scheduler_dict_zhat_z = OmegaConf.to_container(self.hparams.scheduler_config.scheduler_dict
                                                    , resolve=True)
            scheduler_dict_zhat_z["scheduler"] = scheduler_zhat_z
            scheduler_dict_zhat_inv_z_inv = OmegaConf.to_container(self.hparams.scheduler_config.scheduler_dict
                                                    , resolve=True)
            scheduler_dict_zhat_inv_z_inv["scheduler"] = scheduler_zhat_inv_z_inv
            scheduler_dict_zhat_inv_z_spu = OmegaConf.to_container(self.hparams.scheduler_config.scheduler_dict
                                                    , resolve=True)
            scheduler_dict_zhat_inv_z_spu["scheduler"] = scheduler_zhat_inv_z_spu

            return [optimizer_zhat_z, optimizer_zhat_inv_z_inv, optimizer_zhat_inv_z_spu], [scheduler_dict_zhat_z, scheduler_dict_zhat_inv_z_inv, scheduler_dict_zhat_inv_z_spu]
        
        else:
            # no scheduling
            return [optimizer_zhat_z, optimizer_zhat_inv_z_inv, optimizer_zhat_inv_z_spu]

    def training_step(self, batch, batch_idx):

        torch.autograd.set_detect_anomaly(True)
        optimizer_zhat_z, optimizer_zhat_inv_z_inv, optimizer_zhat_inv_z_spu = self.optimizers()

        zhat, z = batch["z_hat"], batch["z"]
        z_hat, z_hat_inv, z_hat_spu = self(zhat, zhat[:, :self.zhat_dim_inv])


        self.toggle_optimizer(optimizer_zhat_z)
        zhat_z_loss = self.loss(z_hat, z)
        self.manual_backward(zhat_z_loss, retain_graph=True)
        optimizer_zhat_z.step()
        optimizer_zhat_z.zero_grad()
        self.untoggle_optimizer(optimizer_zhat_z)
        self.log(f"train_loss", zhat_z_loss.item(), prog_bar=True)
        self.log(f"train_zhat_z_loss", zhat_z_loss.item(), prog_bar=True)

        self.toggle_optimizer(optimizer_zhat_inv_z_inv)
        z_hat_inv_z_inv_loss = self.loss(z_hat_inv, z[:, :self.z_dim_inv])
        self.manual_backward(z_hat_inv_z_inv_loss, retain_graph=True)
        optimizer_zhat_inv_z_inv.step()
        optimizer_zhat_inv_z_inv.zero_grad()
        self.untoggle_optimizer(optimizer_zhat_inv_z_inv)
        self.log(f"train_zhat_inv_z_inv_loss", z_hat_inv_z_inv_loss.item(), prog_bar=True)

        self.toggle_optimizer(optimizer_zhat_inv_z_spu)
        z_hat_inv_z_spu_loss = self.loss(z_hat_spu, z[:, self.z_dim_inv:])
        self.manual_backward(z_hat_inv_z_spu_loss, retain_graph=True)
        optimizer_zhat_inv_z_spu.step()
        optimizer_zhat_inv_z_spu.zero_grad()
        self.untoggle_optimizer(optimizer_zhat_inv_z_spu)
        self.log(f"train_zhat_inv_z_spu_loss", z_hat_inv_z_spu_loss.item(), prog_bar=True)

        # compute the r2 score of the prediction
        r2 = r2_score(z.detach().cpu().numpy(), z_hat.detach().cpu().numpy()) # y_true, y_pred
        self.log(f"train_r2", r2, prog_bar=True)

        r2 = r2_score(z[:, :self.z_dim_inv].detach().cpu().numpy(), z_hat_inv.detach().cpu().numpy())
        self.log(f"train_r2_inv", r2, prog_bar=True)

        r2 = r2_score(z[:, self.z_dim_inv:].detach().cpu().numpy(), z_hat_spu.detach().cpu().numpy())
        self.log(f"train_r2_spu", r2, prog_bar=True)
