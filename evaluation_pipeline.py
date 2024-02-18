from typing import List, Optional
import wandb
import hydra
import os
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase

import slot_based_disentanglement.utils.general as utils

log = utils.get_logger(__name__)

from pathlib import Path
import json


def evaluate(config: DictConfig) -> Optional[float]:
    """Contains the evaluation pipeline.
    Instantiates all PyTorch Lightning objects from configs.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score useful for hyperparameter optimization.
    """

    # Set seed for random number generators in PyTorch, Numpy and Python (random)
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    if not os.path.isabs(config.model.checkpoint_path):
        config.model.checkpoint_path = os.path.join(hydra.utils.get_original_cwd(), config.model.checkpoint_path)
    # Initialize the LIT model (from checkpoint)
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model, _recursive_=False)

    
    # Initialize the LIT data module
    log.info(f"Instantiating data module <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule, _recursive_=False)

    # Initialize LIT callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init LIT loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger, _convert_="partial")
    
    # Send some parameters from configs to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    
    # Test the model
    log.info("Starting testing!")
    model.testing_output_parent_dir = "testing_output"
    Path(model.testing_output_parent_dir).mkdir(exist_ok=True)
    trainer.test(model=model, datamodule=datamodule)
    

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # scores = trainer.callback_metrics
    # log.info("Metrics:")
    # log.info(scores)
    # with open("metrics.json", "w") as f:
    #     f.write(json.dumps(scores))
        