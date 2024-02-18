import torch
import torchvision
import numpy as np
from typing import Callable, Optional
import utils.general as utils
log = utils.get_logger(__name__)
import os


class MixingMultiDomainEncodedDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        num_domains: int = 2,
        **kwargs,
    ):
        super(MixingMultiDomainEncodedDataset, self).__init__()

        self.split = kwargs.get("split", "train")
        self.num_domains = num_domains
        self.path_to_files = kwargs.get("data_dir", None)        
        self.data = torch.load(os.path.join(self.path_to_files, f"encoded_synthetic_mixing_{self.split}.pt"))

        # find the min and max of self.data["z_hat"] and normalize it accordingly
        self.z_hat_min = torch.min(self.data["z_hat"])
        self.z_hat_max = torch.max(self.data["z_hat"])
        self.data["z_hat"] = (((self.data["z_hat"] - self.z_hat_min) / (self.z_hat_max - self.z_hat_min)) - 0.5) * 2
        
        self.z_min = torch.min(self.data["z"])
        self.z_max = torch.max(self.data["z"])
        # self.data["z"] = (((self.data["z"] - self.z_min) / (self.z_max - self.z_min)) - 0.5) * 2

    def __getitem__(self, idx):
        return {"x": self.data["z_hat"][idx], "z": self.data["z"][idx], "domain": self.data["domain"][idx]}

    def __len__(self):
        return len(self.data["z"])
