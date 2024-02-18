
To reproduce, please run the following commands after installing and activating the environments with the `requirements.txt`


For linear mixing (and different penalties), the following runs all the combinations presented in the paper with 5 seeds:

```







```


For these runs, first the data should be generated and stage 1 (reconstruction) is carried out until convergence. Below is a sample command for a polynomial of degree 3, with latent dimension = 14

```
```
You can change `datamodule.dataset.correlated_z`, and `datamodule.dataset.corr_prob` to achieve different SCMs (independent/dynamic).
Then use the path of the resulting run (where encoded dataset from stage 1 will be stored) for stage 2 as follows:
```
python run_training.py ckpt_path=null model=mixing_md_encoded_autoencoder datamodule=mixing_encoded datamodule.batch_size=1024 model.penalty_criterion.minmax=1. model.penalty_criterion.mmd=0. ~callbacks.visualization_callback logger.wandb.tags=["poly-uniform","minmax","p3"] run_path="'path/to/above/run/'" seed=1235,4256,49685,7383,9271 --multirun
```
You can use any combination of penalties by switching `model.penalty_criterion.minmax` and `model.penalty_criterion.mmd` to `0.`, `1.`. 


Similarly, for these runs, first the data should be generated and stage 1 (reconstruction) is carried out until convergence. Below is a sample command for the balls dataset with 2 balls, one invariant and one spurious, with 16 domains.

```
```
You can change `datamodule.dataset.correlated_z`, and `datamodule.dataset.corr_prob` to achieve different SCMs (independent/dynamic).
Then use the path of the resulting run (where encoded dataset from stage 1 will be stored) for stage 2 as follows:
```
python run_training.py trainer.accelerator="cpu" trainer.devices="auto" model.optimizer.lr=0.001 datamodule=balls_encoded datamodule.batch_size=1024 model=balls_md_encoded_autoencoder model.penalty_weight=1.0 logger.wandb.tags=["balls-uniform","minmax"] ~callbacks.early_stopping ~callbacks.visualization_callback run_path="path/to/run/" model.z_dim=128 model.hinge_loss_weight=0.0 model.z_dim_invariant_fraction=0.5 model/autoencoder=mlp_ae_balls ckpt_path=null model.penalty_criterion.mmd=0. model.penalty_criterion.minmax=1. model.save_encoded_data=False seed=1235,4256,49685,7383,9271 --multirun
```
You can use any combination of penalties by switching `model.penalty_criterion.minmax` and `model.penalty_criterion.mmd` to `0.`, `1.`. 


This project is licensed under CC-BY-NC as seen in License file