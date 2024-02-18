# Multi-Domain Causal Representation Learning

To reproduce, please run the following commands after installing and activating the environments with the `requirements.txt`


## Linear Mixing
For linear mixing (and different penalties), the following runs all the combinations presented in the paper with 5 seeds:

```
# minmax penalty, independent SCM
# python run_training.py ckpt_path=null model=mixing_synthetic model.penalty_criterion.minmax=1. model.penalty_criterion.mmd=0. datamodule=mixing datamodule.batch_size=1024 datamodule.dataset.linear=True datamodule.dataset.z_dim=8,16,32,64 datamodule.dataset.num_domains=2,4,8,16 ~callbacks.visualization_callback logger.wandb.tags=["linear-uniform","minmax"] datamodule.dataset.correlated_z=False seed=1235,4256,49685,7383,9271 --multirun

# minmax penalty, dynamic SCM
# python run_training.py ckpt_path=null model=mixing_synthetic model.penalty_criterion.minmax=1. model.penalty_criterion.mmd=0. datamodule=mixing datamodule.batch_size=1024 datamodule.dataset.linear=True datamodule.dataset.z_dim=8,16,32,64 datamodule.dataset.num_domains=2,4,8,16 ~callbacks.visualization_callback logger.wandb.tags=["linear-corr","minmax"] datamodule.dataset.correlated_z=True datamodule.dataset.corr_prob=0.5 seed=1235,4256,49685,7383,9271 --multirun

# mmd penalty, independent SCM
# python run_training.py ckpt_path=null model=mixing_synthetic model.penalty_criterion.minmax=0. model.penalty_criterion.mmd=1. model.mmd_loss.fix_sigma=null datamodule=mixing datamodule.batch_size=1024 datamodule.dataset.linear=True datamodule.dataset.z_dim=8,16,32,64 datamodule.dataset.num_domains=2,4,8,16 ~callbacks.visualization_callback logger.wandb.tags=["linear-uniform","mmd"] datamodule.dataset.correlated_z=False datamodule.dataset.corr_prob=0.0 seed=1235,4256,49685,7383,9271 --multirun

# mmd penalty, dynamic SCM
# python run_training.py ckpt_path=null model=mixing_synthetic model.penalty_criterion.minmax=0. model.penalty_criterion.mmd=1. model.mmd_loss.fix_sigma=null datamodule=mixing datamodule.batch_size=1024 datamodule.dataset.linear=True datamodule.dataset.z_dim=8,16,32,64 datamodule.dataset.num_domains=2,4,8,16 ~callbacks.visualization_callback logger.wandb.tags=["linear-corr","mmd"] datamodule.dataset.correlated_z=True datamodule.dataset.corr_prob=0.5 seed=1235,4256,49685,7383,9271 --multirun

# mmd + minmax, independent SCM
# python run_training.py ckpt_path=null model=mixing_synthetic model.penalty_criterion.minmax=1. model.penalty_criterion.mmd=1. model.mmd_loss.fix_sigma=null datamodule=mixing datamodule.batch_size=1024 datamodule.dataset.linear=True datamodule.dataset.z_dim=8,16,32,64 datamodule.dataset.num_domains=2,4,8,16 ~callbacks.visualization_callback logger.wandb.tags=["linear-uniform","minmax-mmd"] datamodule.dataset.correlated_z=False seed=1235,4256,49685,7383,9271 --multirun

# mmd + minmax, dynamic SCM
# python run_training.py ckpt_path=null model=mixing_synthetic model.penalty_criterion.minmax=1. model.penalty_criterion.mmd=1. model.mmd_loss.fix_sigma=null datamodule=mixing datamodule.batch_size=1024 datamodule.dataset.linear=True datamodule.dataset.z_dim=8,16,32,64 datamodule.dataset.num_domains=2,4,8,16 ~callbacks.visualization_callback logger.wandb.tags=["linear-corr","minmax-mmd"] datamodule.dataset.correlated_z=True datamodule.dataset.corr_prob=0.5 seed=1235,4256,49685,7383,9271 --multirun


```

## Polynomial Mixing

For these runs, first the data should be generated and stage 1 (reconstruction) is carried out until convergence. Below is a sample command for a polynomial of degree 3, with latent dimension = 14

```
# python run_training.py ckpt_path=null model=mixing_synthetic model/autoencoder=poly_ae model.penalty_criterion.minmax=0. model.penalty_criterion.mmd=0. datamodule=mixing datamodule.batch_size=1024 datamodule.dataset.linear=False datamodule.dataset.non_linearity=polynomial datamodule.dataset.polynomial_degree=3 datamodule.dataset.z_dim=14 datamodule.dataset.x_dim=200 datamodule.dataset.num_domains=16 ~callbacks.visualization_callback logger.wandb.tags=["poly-stage-1-uniform"] model.save_encoded_data=True datamodule.dataset.correlated_z=False datamodule.dataset.corr_prob=0.0 seed=6739
```
You can change `datamodule.dataset.correlated_z`, and `datamodule.dataset.corr_prob` to achieve different SCMs (independent/dynamic).
Then use the path of the resulting run (where encoded dataset from stage 1 will be stored) for stage 2 as follows:
```
python run_training.py ckpt_path=null model=mixing_md_encoded_autoencoder datamodule=mixing_encoded datamodule.batch_size=1024 model.penalty_criterion.minmax=1. model.penalty_criterion.mmd=0. ~callbacks.visualization_callback logger.wandb.tags=["poly-uniform","minmax","p3"] run_path="'path/to/above/run/'" seed=1235,4256,49685,7383,9271 --multirun
```
You can use any combination of penalties by switching `model.penalty_criterion.minmax` and `model.penalty_criterion.mmd` to `0.`, `1.`. 

## Balls Image Dataset

Similarly, for these runs, first the data should be generated and stage 1 (reconstruction) is carried out until convergence. Below is a sample command for the balls dataset with 2 balls, one invariant and one spurious, with 16 domains.

```
# python run_training.py trainer.accelerator='gpu' trainer.devices="auto" ckpt_path=null datamodule=md_balls model=balls model.z_dim=128 model/autoencoder=resnet18_ae_balls ~callbacks.early_stopping datamodule.save_dataset=True datamodule.load_dataset=False datamodule.dataset.correlated_z=False datamodule.dataset.corr_prob=0.0 datamodule.dataset.num_domains=16 datamodule.dataset.invariant_low='[0.5,0.5]' datamodule.dataset.invariant_high='[0.9,0.9]'
```
You can change `datamodule.dataset.correlated_z`, and `datamodule.dataset.corr_prob` to achieve different SCMs (independent/dynamic).
Then use the path of the resulting run (where encoded dataset from stage 1 will be stored) for stage 2 as follows:
```
python run_training.py trainer.accelerator="cpu" trainer.devices="auto" model.optimizer.lr=0.001 datamodule=balls_encoded datamodule.batch_size=1024 model=balls_md_encoded_autoencoder model.penalty_weight=1.0 logger.wandb.tags=["balls-uniform","minmax"] ~callbacks.early_stopping ~callbacks.visualization_callback run_path="path/to/run/" model.z_dim=128 model.hinge_loss_weight=0.0 model.z_dim_invariant_fraction=0.5 model/autoencoder=mlp_ae_balls ckpt_path=null model.penalty_criterion.mmd=0. model.penalty_criterion.minmax=1. model.save_encoded_data=False seed=1235,4256,49685,7383,9271 --multirun
```
You can use any combination of penalties by switching `model.penalty_criterion.minmax` and `model.penalty_criterion.mmd` to `0.`, `1.`. 

