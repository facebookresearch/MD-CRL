import pytorch_lightning as pl
import hydra
import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        try:
            resnet18 = kwargs['encoder_layers']['resnet18']
        except:
            import torchvision.models as models
            resnet18 = models.resnet18(pretrained=False)  # Create an empty ResNet18 model
            # resnet18.load_state_dict(torch.load("resnet18-f37072fd.pth"))  # Load weights from the checkpoint

        mlp_layers = torch.nn.Sequential(
            *[layer_config for _, layer_config in kwargs['encoder_layers']['mlp_layers'].items()]
        )
        self.layers = torch.nn.Sequential(*list(resnet18.children())[:-1], *mlp_layers)
        # for param in resnet18.parameters():
        #     param.requires_grad = False


    def forward(self, x):
        
        # input `x` or `image` has shape: [batch_size, num_channels, width, height].
        # the output is of dimensions [batch_size, latent_dim]
        x = self.layers(x)
        # # if x is 2D, we need to add the extra dimensions to make it [batch_size, latent_dim, 1, 1]
        # if len(x.shape) == 2:
        #     x = torch.unsqueeze(torch.unsqueeze(x, -1), -1)
        return x


class Decoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.width = kwargs['width']
        self.height = kwargs['height']
        self.num_channels = kwargs['num_channels']
        
        # note that encoder layers are instantiated recursively by hydra, so we only need to connect them
        # by nn.Sequential
        self.mlp_layers = torch.nn.Sequential(
            *[layer_config for _, layer_config in kwargs['mlp_layers'].items()]
        )

        self.deconv_layers = torch.nn.Sequential(
            *[layer_config for _, layer_config in kwargs['decoder_layers'].items()]
        )
        self.layers = torch.nn.Sequential(*self.mlp_layers, *self.deconv_layers)
                
    def forward(self, x):

        # `x` has shape: [batch_size, latent_dim].

        x = self.mlp_layers(x) # [batch_size, 64*4*4]
        x = x.view(x.size(0), 64, 4, 4) # [batch_size, 64, 4, 4]
        x = self.deconv_layers(x) # [batch_size, num_channels, width, height]
        return x
        

class ResNetAE(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__()

        self.encoder = hydra.utils.instantiate(kwargs['encoder_cnn'])
        self.decoder = hydra.utils.instantiate(kwargs['decoder_cnn'])


    def forward(self, image):
        # `image` has shape: [batch_size, num_channels, width, height].
        z = self.encoder(image)
        recons = self.decoder(z)
        return torch.reshape(z, (z.shape[0], -1)), recons

