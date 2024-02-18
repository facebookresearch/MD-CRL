import torch
import os
import numpy as np
import math
from typing import Callable, Optional
import utils.general as utils
log = utils.get_logger(__name__)
from tqdm import tqdm

    
class MDMixingPickleable(torch.utils.data.Dataset):
    """
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        data = None,
        transform: Optional[Callable] = None,
        **kwargs,
    ):
        super(MDMixingPickleable, self).__init__()

        self.data = data
        self.length = dataset.__len__()
        # extract all attributes of dataset as the attributes of this new dataset
        for attr in dir(dataset):
            if not attr.startswith("__") and not attr == "data" and not attr.startswith("_generate") and not attr == "pickleable_dataset" and not attr == "_mixing_G" and not attr == "G" and not attr == "_correlate_z":
                # print(attr)
                setattr(self, attr, getattr(dataset, attr))

    def __getitem__(self, idx):
        return {"x": self.data[0][idx], "z": self.data[1][idx], "domain": self.data[2][idx]}

    def __len__(self) -> int:
        return self.length


