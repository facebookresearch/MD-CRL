# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import os.path as path
import hydra
import numpy as np
from datamodule.augmentations import ALL_AUGMENTATIONS

AUGMENTATIONS = {k: lambda x: v(x, order=1) for k, v in ALL_AUGMENTATIONS.items()}

import utils.general as utils
log = utils.get_logger(__name__)
    

class EncodedDataModule(LightningDataModule):
    def __init__(self
                 , seed: int = 1234
                 , batch_size: int= 128
                 , **kwargs):
        
        super().__init__()
        
        # So all passed parameters are accessible through self.hparams
        self.save_hyperparameters(logger=False)
        self.dataset_parameters = self.hparams.dataset["dataset_parameters"]
        self.dataset_name = self.hparams["dataset_name"]
        self.datamodule_name = self.hparams["datamodule_name"]
        
        self.seed = seed
        self.dirname = os.path.dirname(__file__)
        self.path_to_files = self.hparams["data_dir"]
        print("datamodule init successful")

    def prepare_data(self):
        
        import time
        start_time = time.perf_counter()
        log.info(f"Loading the train dataset files from {self.path_to_files}")
        self.train_dataset = hydra.utils.instantiate(self.dataset_parameters["train"]["dataset"])
        log.info(f"Loading the training dataset files took {time.perf_counter() - start_time} seconds.")
        
        start_time = time.perf_counter()
        self.valid_dataset = hydra.utils.instantiate(self.dataset_parameters["valid"]["dataset"])
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
