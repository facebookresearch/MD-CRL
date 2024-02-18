import torch
from torch.nn import functional as F
from .base_pl import BasePl
import hydra
from omegaconf import OmegaConf
import wandb
import os
from utils.disentanglement_utils import linear_disentanglement, permutation_disentanglement
import utils.general as utils
log = utils.get_logger(__name__)


class AutoencoderPL(BasePl):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        # self.model = ResNet18Autoencoder()
        self.model = hydra.utils.instantiate(self.hparams.autoencoder, _recursive_=False)
        if self.hparams.get("autoencoder_ckpt_path", None) is not None:    
            ckpt_path = self.hparams["autoencoder_ckpt_path"]
            # only load the weights, i.e. HPs should be overwritten from the passed config
            # b/c maybe the ckpt has num_slots=7, but we want to test it w/ num_slots=12
            # NOTE: NEVER DO self.model = self.model.load_state_dict(...), raises _IncompatibleKey error
            self.model.load_state_dict(torch.load(ckpt_path))
            self.hparams.pop("autoencoder_ckpt_path") # we don't want this to be save with the ckpt, sicne it will raise key errors when we further train the model
                                                  # and load it for evaluation.

        # remove the state_dict_randomstring.ckpt to avoid cluttering the space
        import os
        import glob
        state_dicts_list = glob.glob('./state_dict_*.pth')
        # for state_dict_ckpt in state_dicts_list:
        #     try:
        #         os.remove(state_dict_ckpt)
        #     except:
        #         print("Error while deleting file: ", state_dict_ckpt)

        # freeze the parameters of encoder if needed
        if self.hparams.autoencoder_freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        else: # if the flag is set to true we should correct the requires_grad flags, i.e. we might
              # initially freeze it for some time, but then decide to let it finetune.
            for param in self.model.parameters():
                param.requires_grad = True
        
        self.training_step_outputs = []
        self.validation_step_outputs = []


    def forward(self, x):
        return self.model(x)

    def loss(self, images, recons, z):

        # images, recons: [batch_size, num_channels, width, height]
        reconstruction_loss = F.mse_loss(recons.permute(0, 2, 3, 1), images.permute(0, 2, 3, 1), reduction="mean")
        penalty_loss = 0.
        loss = reconstruction_loss + penalty_loss
        return loss, reconstruction_loss, penalty_loss

    def training_step(self, train_batch, batch_idx):

        # images: [batch_size, num_channels, width, height]
        images, labels = train_batch["image"], train_batch["label"]

        # z: [batch_size, latent_dim]
        # recons: [batch_size, num_channels, width, height]
        z, recons = self(images)

        loss, reconstruction_loss, penalty_loss = self.loss(images, recons, z)
        self.log(f"train_reconstruction_loss", reconstruction_loss.item())
        # self.log(f"penalty_loss", penalty_loss.item())
        self.log(f"train_loss", loss.item())

        self.training_step_outputs.append({"z":z, "label":labels, "domain": train_batch["domain"], "color": train_batch["color"]})

        return loss

    def validation_step(self, valid_batch, batch_idx):

        # images: [batch_size, num_channels, width, height]
        images, labels = valid_batch["image"], valid_batch["label"]

        # z: [batch_size, latent_dim]
        # recons: [batch_size, num_channels, width, height]
        z, recons = self(images)

        # we have the set of labels and latents. We want to train a classifier to predict the labels from latents
        # using multinomial logistic regression using sklearn
        # import sklearn
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.metrics import accuracy_score, r2_score
        # fit a multinomial logistic regression from z to labels
        clf = LogisticRegression(random_state=0, max_iter=1000).fit(z.detach().cpu().numpy(), labels.detach().cpu().numpy())
        # predict the labels from z
        pred_labels = clf.predict(z.detach().cpu().numpy())
        # compute the accuracy
        accuracy = accuracy_score(labels.detach().cpu().numpy(), pred_labels)

        
        self.log(f"val_digits_accuracy", accuracy, prog_bar=True)
        loss, reconstruction_loss, penalty_loss = self.loss(images, recons, z)
        self.log(f"val_reconstruction_loss", reconstruction_loss.item())
        # self.log(f"valid_penalty_loss", penalty_loss.item())
        self.log(f"val_loss", loss.item())
        
        # fit a linear regression from z to colours
        colors = valid_batch["color"]
        clf = LinearRegression().fit(z.detach().cpu().numpy(), colors.detach().cpu().numpy())
        pred_colors = clf.predict(z.detach().cpu().numpy())
        r2 = r2_score(colors.detach().cpu().numpy(), pred_colors)
        self.log(f"colors_r2", r2, prog_bar=True)

        self.validation_step_outputs.append({"z":z, "label":labels, "domain": valid_batch["domain"], "color": valid_batch["color"]})

        return {"loss": loss, "pred_z": z}

    def on_train_epoch_end(self):

        # at the end of each validation epoch, we want to pass the whole dataset through the model
        # and save the outputs of the encoder as a new dataset
        # we also want to save the labels, domains, and colours
        # of the dataset.
        
        # instantiate the new data with the same keys as the original dataset with zeros tensors
        new_data = dict.fromkeys(["z", "label", "domain", "color"])
        key_dims = {"z": self.hparams.z_dim, "label": 1, "domain": 1, "color": 3}
        for key in new_data.keys():
            new_data[key] = torch.zeros((len(self.trainer.datamodule.train_dataset), key_dims[key]))
        
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
        log.info(f"Saving the encoded training dataset of length {len(new_data['z'])} at: {os.getcwd()}")
        torch.save(new_data, os.path.join(os.getcwd(), f"encoded_img_{self.trainer.datamodule.datamodule_name}_train.pt"))
        self.training_step_outputs.clear()

        return
    
    def on_validation_epoch_end(self):
        
        # instantiate the new data with the same keys as the original dataset with zeros tensors
        new_data = dict.fromkeys(["z", "label", "domain", "color"])
        key_dims = {"z": self.hparams.z_dim, "label": 1, "domain": 1, "color": 3}
        for key in new_data.keys():
            new_data[key] = torch.zeros((len(self.trainer.datamodule.valid_dataset), key_dims[key]))
        
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
        log.info(f"Saving the encoded validation dataset of length {len(new_data['z'])} at: {os.getcwd()}")
        torch.save(new_data, os.path.join(os.getcwd(), f"encoded_img_{self.trainer.datamodule.datamodule_name}_valid.pt"))
        self.validation_step_outputs.clear()

        return


import torch.nn as nn
import torchvision.models as models

# Autoencoder with ResNet18 Encoder
class ResNet18Autoencoder(nn.Module):
    def __init__(self, **kwargs):
        super(ResNet18Autoencoder, self).__init__()
        
        num_channels = 3
        # Load pretrained ResNet18
        resnet18 = models.resnet18(pretrained=True)
        # Modify the last fully connected layer to output 64 features
        z_dim = kwargs.get("z_dim", 64)
        resnet18.fc = nn.Linear(512, z_dim)
        self.encoder = resnet18 # nn.Sequential(*list(resnet18.children())[:-2])  # Exclude the last two layers

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 32, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 4, kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, num_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # # nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            # # nn.ReLU(),
            # nn.ConvTranspose2d(32, 16, kernel_size=2, stride=1, padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            # nn.Sigmoid()  # Output range [0, 1] for images
        )


    def forward(self, x):

        # x: [batch_size, num_channels, width, height]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded.view(encoded.size(0), 64, 1, 1))
        return encoded, decoded
