# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch.utils.data
import torch
import os
import numpy as np
from typing import Callable, Optional


class MNISTBase(torch.utils.data.Dataset):
    """
    MNIST dataset with regular images.
    """
    def __init__(self, transform: Optional[Callable] = None
                 , num_samples: int = 20000
                 , path: str = "/network/datasets/torchvision" 
                 , **kwargs): # type: ignore
        super(MNISTBase, self).__init__()

        if transform is None:
            def transform(x):
                return x

        self.transform = transform
        self.data = []

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {len(self)}"]

        if hasattr(self, "transforms") and self.transform is not None:
            body += [repr(self.transform)]
        lines = [head] + [" " + line for line in body]
        return "\n".join(lines)

    def _generate_data(self):
        raise NotImplementedError