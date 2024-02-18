from typing import List, Optional
import wandb
import hydra
import os
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
# from pytorch_lightning.loggers import LightningLoggerBase

import utils.general as utils

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from configs.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score useful for hyperparameter optimization.
    """

    # Set seed for random number generators in PyTorch, Numpy and Python (random)
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Initialize the LIT model
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
    logger = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

#     wandb.watch(
#         model,
#         criterion=None,
#         log="gradients",
#         log_freq=1000,
#         idx=None,
#         log_graph=(False),
#     )

        
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


    # Load a ckpt if specified, o.w. it will be None and will have no effect on the trainer.
    if config.ckpt_path is not None:
        log.info(f"Resuming from the following ckpt:{config.ckpt_path}")
        # modify ckpt keys for optimizer state and lr schedulers so it doesn't follow the ckpt values.
        import torch
        with open(config.ckpt_path, "rb") as f:
            ckpt = torch.load(f)
            ckpt["optimizer_states"] = []
#             for key in ckpt["optimizer_states"][0].keys():
#                 ckpt["optimizer_states"][0][key] = []
            ckpt["lr_schedulers"] = []
        with open(config.ckpt_path, "wb") as f:
            torch.save(ckpt, f)
        
    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=config.ckpt_path)

    # Print the path to best checkpoint
    if not config.trainer.get("fast_dev_run"):
        log.info(f"Best checkpoint at: {trainer.checkpoint_callback.best_model_path}")
        # log.info(f"Best checkpoint Directory: {os.path.dirname(trainer.checkpoint_callback.best_model_path)}")
        # log.info(f"Best checkpoint filename: {os.path.basename(trainer.checkpoint_callback.best_model_path)}")
        with open("best_ckpt_path.txt", "w") as f:
            f.write(os.path.basename(trainer.checkpoint_callback.best_model_path))

    # Test the model
    if config.get("test"):
        ckpt_path = "best"  # Use the best checkpoint from the previous trainer.fit() call
        _model = None
        if config.get("ckpt_path"):
            ckpt_path = config.get("ckpt_path")  # Use the checkpoint passed in the config
        elif not config.get("train") or config.trainer.get("fast_dev_run"):
            _model = model
            ckpt_path = None  # Use the passed model as it is

        # changing latent matching to constrained LP for evaluation
        # _model.latent_matching = "constrained_lp"
        log.info("Starting testing!")
        trainer.test(model=_model, datamodule=datamodule, ckpt_path=ckpt_path)

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

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # # Get metric score for hyperparameter optimization
    # optimized_metric = config.get("optimized_metric")
    # if optimized_metric and optimized_metric not in trainer.callback_metrics:
    #     raise Exception(
    #         "Metric for hyperparameter optimization not found! "
    #         "Make sure the `optimized_metric` in `hparams_search` config is correct!"
    #     )

    # score = trainer.callback_metrics.get(optimized_metric)
    # return score
