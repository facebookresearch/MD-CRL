# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pytorch_lightning.callbacks import Callback
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as Image, ImageEnhance


class VisualizationLoggerCallback(Callback):
    

    def __init__(self, visualize_every=100, n_samples=3, **kwargs):
        
        self.visualize_every = visualize_every
        self.n_samples = n_samples
    
    def on_fit_start(self, trainer, pl_module):
        
        self.datamodule = trainer.datamodule
        self.model = pl_module.model
        
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % (self.visualize_every+1) == 0:

            try:
                renormalize = self.datamodule.train_dataset.renormalize()
            except AttributeError:
                renormalize = self.datamodule.train_dataset.dataset.renormalize()
            except:
                renormalize = self.datamodule.train_dataset.dataset.dataset.renormalize()
            
            pl_module.eval()
            with torch.no_grad():
            
                # images, recons: [batch_size, num_channels, width, height]
                images = batch["image"][:self.n_samples]
                # import code
                # code.interact(local=locals())
                try:
                    from models.modules.mlp_ae import FCAE
                    if isinstance(self.model, FCAE):
                        images_ = images.view(images.size(0), -1).clone() # flatten the input
                        z, recons = self.model(images_)
                        recons = recons.view(recons.size(0), 3, 28, 28) # reshape the output
                    else:
                        z, recons = self.model(images)
                except:
                    z, recons, _, _ = self.model(images)

                images = images.permute(0, 2, 3, 1)
                recons = recons.permute(0, 2, 3, 1)
                num_channels = images.shape[1]
                # renormalize image and recons?

                plt.cla()
                plt.close('all')
                fig, ax = plt.subplots(2, self.n_samples, figsize=(10, 4))
                for idx in range(self.n_samples):
                    if num_channels == 1:
                        image = images[idx].squeeze().cpu().numpy() # [width, height]
                        recon_ = recons[idx].squeeze().cpu().numpy() # [width, height]
                        color_map = "gray"
                    else:
                        image = images[idx].cpu().numpy() # [width, height, num_channels]
                        recon_ = recons[idx].cpu().numpy() # [width, height, num_channels]
                        image = self.clamp(renormalize(image))
                        recon_ = self.clamp(renormalize(recon_))
                        # print(f"image min, max: {image.min()}, {image.max()}")
                        # print(f"recon_ min, max: {recon_.min()}, {recon_.max()}")
                        color_map = None


                    # Visualize.
                    if color_map == "gray":
                        ax[0,idx].imshow(image, cmap=color_map)
                        ax[1,idx].imshow((recon_ * 255).astype(np.uint8), vmin=0, vmax=255, cmap=color_map)
                    else:
                        ax[0,idx].imshow(image, vmin=0, vmax=1)
                        ax[1,idx].imshow(recon_, vmin=0, vmax=1)
                        # ax[0,idx].imshow(recon_)
                        
                    ax[0,idx].set_title('Image')
                    ax[1,idx].set_title('Recon.')
                    ax[0,idx].grid(False)
                    ax[0,idx].axis('off')
                    ax[1,idx].grid(False)
                    ax[1,idx].axis('off')

                wandb.log({f"Val Reconstructions": fig})


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % (self.visualize_every+1) == 0:

            try:
                renormalize = self.datamodule.train_dataset.renormalize()
            except AttributeError:
                renormalize = self.datamodule.train_dataset.dataset.renormalize()
            except:
                renormalize = self.datamodule.train_dataset.dataset.dataset.renormalize()
            
            pl_module.eval()
            with torch.no_grad():
            
                # images, recons: [batch_size, num_channels, width, height]
                images = batch["image"][:self.n_samples]
                try:
                    from models.modules.mlp_ae import FCAE
                    if isinstance(self.model, FCAE):
                        images_ = images.view(images.size(0), -1).clone() # flatten the input
                        z, recons = self.model(images_)
                        recons = recons.view(recons.size(0), 3, 28, 28) # reshape the output
                    else:
                        z, recons = self.model(images)
                except:
                    z, recons, _, _ = self.model(images)

                images = images.permute(0, 2, 3, 1)
                recons = recons.permute(0, 2, 3, 1)
                num_channels = images.shape[1]
                # renormalize image and recons?

                plt.cla()
                plt.close('all')
                fig, ax = plt.subplots(2, self.n_samples, figsize=(10, 4))
                for idx in range(self.n_samples):
                    if num_channels == 1:
                        image = images[idx].squeeze().cpu().numpy() # [width, height]
                        recon_ = recons[idx].squeeze().cpu().numpy() # [width, height]
                        color_map = "gray"
                    else:
                        image = images[idx].cpu().numpy() # [width, height, num_channels]
                        recon_ = recons[idx].cpu().numpy() # [width, height, num_channels]
                        image = self.clamp(renormalize(image))
                        recon_ = self.clamp(renormalize(recon_))
                        # print(f"image min, max: {image.min()}, {image.max()}")
                        # print(f"recon_ min, max: {recon_.min()}, {recon_.max()}")
                        color_map = None


                    # Visualize.
                    if color_map == "gray":
                        ax[0,idx].imshow(image, cmap=color_map)
                        ax[1,idx].imshow((recon_ * 255).astype(np.uint8), vmin=0, vmax=255, cmap=color_map)
                    else:
                        ax[0,idx].imshow(image, vmin=0, vmax=1)
                        ax[1,idx].imshow(recon_, vmin=0, vmax=1)
                        # ax[0,idx].imshow(recon_)
                        
                    ax[0,idx].set_title('Image')
                    ax[1,idx].set_title('Recon.')
                    ax[0,idx].grid(False)
                    ax[0,idx].axis('off')
                    ax[1,idx].grid(False)
                    ax[1,idx].axis('off')

                wandb.log({f"Train Reconstructions": fig})

    # a function to clamps the values of a numpy array between 0,1
    def clamp(self, x):
        return np.minimum(np.maximum(x, 0), 1)

            
