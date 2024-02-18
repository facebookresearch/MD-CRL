import os
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import os.path as path
import hydra
import numpy as np
import utils.general as utils
log = utils.get_logger(__name__)
    

class XZDataModule(LightningDataModule):
    def __init__(self
                 , seed: int = 1234
                 , batch_size: int= 128
                 , **kwargs):
        
        super().__init__()
        
        # So all passed parameters are accessible through self.hparams
        self.save_hyperparameters(logger=False)
        
        self.seed = seed
        self.dirname = os.path.dirname(__file__)
        self.path_to_files = self.hparams["data_dir"]
        self.dataset_parameters = self.hparams["dataset_parameters"]

    def prepare_data(self):
        
        import time
        start_time = time.perf_counter()
        log.info(f"Loading the train dataset files from {self.path_to_files}")
        self.train_dataset = torch.load(os.path.join(self.path_to_files, "double_encoded_img_balls_encoded_train.pt"))
        self.train_dataset = XZDataset(self.train_dataset)
        log.info(f"Loading the training dataset files took {time.perf_counter() - start_time} seconds.")
        
        start_time = time.perf_counter()
        self.valid_dataset = torch.load(os.path.join(self.path_to_files, "double_encoded_img_balls_encoded_valid.pt"))
        self.valid_dataset = XZDataset(self.valid_dataset)
        log.info(f"Loading the validation dataset files took {time.perf_counter() - start_time} seconds.")
        
        self.test_dataset = self.valid_dataset

    def setup(self, stage=None):
        pass
        

    def train_dataloader(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        return DataLoader(
            dataset=self.train_dataset,
            **self.dataset_parameters['train']['dataloader'],
            generator=g,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset, **self.dataset_parameters['valid']['dataloader']
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset, **self.dataset_parameters['test']['dataloader']
        )


class XZDataset(torch.utils.data.Dataset):
    """
    This class instantiates a torch.utils.data.Dataset object.
    """

    def __init__(
        self,
        data,
        **kwargs,
    ):
        super(XZDataset, self).__init__()
        self.data = data


    def __len__(self) -> int:
        return len(self.data["z"])

    def __getitem__(self, idx):
        return {"z_hat": self.data["z_hat"][idx].float(), "z": self.data["z"][idx].float()}
