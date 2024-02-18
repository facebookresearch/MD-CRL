import torch
import os
import numpy as np
import math
from typing import Callable, Optional
import colorsys
import utils.general as utils
log = utils.get_logger(__name__)
from tqdm import tqdm
log_ = False

# HSV colours
COLOURS_ = [
    # [0.05, 0.6, 0.6],
    [0.15, 0.6, 0.6],
    # [0.25, 0.6, 0.6],
    # [0.35, 0.6, 0.6],
    [0.45, 0.6, 0.6],
    # [0.55, 0.6, 0.6],
    # [0.65, 0.6, 0.6],
    [0.75, 0.6, 0.6],
    # [0.85, 0.6, 0.6],
    # [0.95, 0.6, 0.6],
]

SHAPES_ = [
    "circle",
    "square",
    "triangle",
    "heart"
]

PROPERTIES_ = [
    "x",
    "y",
    "c",
    "s",
    "l",
    "p",
]

    
class MDBallsPickleable(torch.utils.data.Dataset):
    """
    """

    def __init__(
        self,
        data = None,
        transform: Optional[Callable] = None,
        **kwargs,
    ):
        super(MDBallsPickleable, self).__init__()

        self.data = data
        self.transform = transform
        # convert all kwargs to attributes
        for k, v in kwargs.items():
            setattr(self, k, v)


    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)

    def renormalize(self):
        for t in self.transform.transforms:
            if t.__class__.__name__ == "Standardize":
                """Renormalize from [-1, 1] to [0, 1]."""
                return lambda x: x / 2.0 + 0.5
        # return lambda x: x
        # return lambda x: x * self.std_ + self.mean_
        # return lambda x: x * (self.max_ - self.min_) + self.min_
