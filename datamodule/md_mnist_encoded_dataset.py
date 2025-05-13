# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torchvision
import numpy as np
from typing import Callable, Optional
import utils.general as utils
log = utils.get_logger(__name__)
import os


class MNISTMultiDomainEncodedDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        num_domains: int = 2,
        **kwargs,
    ):
        super(MNISTMultiDomainEncodedDataset, self).__init__()

        self.split = kwargs.get("split", "train")
        self.num_domains = num_domains
        self.generation_strategy = kwargs.get("generation_strategy", "auto")
        self.path_to_files = kwargs.get("data_dir", None)
        self.data = torch.load(os.path.join(self.path_to_files, f"encoded_img_multi_domain_mnist_{self.num_domains}_{self.split}.pt"))
        self.normalize = kwargs.get("normalize", True)

        # find the min and max of self.data["z_hat"] and normalize it accordingly
        self.z_hat_min = torch.min(self.data["z_hat"])
        self.z_hat_max = torch.max(self.data["z_hat"])
        if self.normalize:
            self.data["z_hat"] = (self.data["z_hat"] - self.z_hat_min) / (self.z_hat_max - self.z_hat_min)

    def __getitem__(self, idx):
        return {"x": self.data["z_hat"][idx], "label": self.data["label"][idx], "domain": self.data["domain"][idx], "color": self.data["color"][idx]}

    def __len__(self):
        return len(self.data["z_hat"])
