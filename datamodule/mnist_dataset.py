from .abstract_mnist_dataset import MNISTBase
import torch
import torchvision
from torchvision import transforms
import numpy as np
from typing import Callable, Optional
import utils.general as utils
log = utils.get_logger(__name__)
from tqdm import tqdm
log_ = False

class MNISTRegularDataset(MNISTBase):
    """
    This class instantiates a torch.utils.data.Dataset object.
    """

    def __init__(
        self,
        transform: Optional[Callable] = None, # type: ignore
        num_samples: int = 20000,
        path: str = "/network/datasets/torchvision",
        **kwargs,
    ):
        super(MNISTRegularDataset, self).__init__(transform
                                            , num_samples
                                            , path
                                            ,**kwargs
                                            )
        if transform is None:
            def transform(x):
                return x

        self.transform = transform
        self.path = path
        self.split = kwargs.get("split", "train")
        self.data = self._generate_data()

    def _generate_data(self):
        if self.split == "train":
            data = torchvision.datasets.MNIST(self.path, True, transform=self.transform)
        else:
            data = torchvision.datasets.MNIST(self.path, False, transform=self.transform)
        return data

    def __getitem__(self, idx):

        return {"image": self.data[idx][0], "label": self.data[idx][1]}
