from utils import hydra_custom_resolvers
import hydra
from omegaconf import DictConfig

# python run_training.py training=<evaluation_config> ckpt_path=<path_to_ckpt_to_evaluate> run_name=<run_name>


@hydra.main(config_path="configs", config_name="train_root", version_base="1.3.2")
def main(hydra_config: DictConfig):
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934

    import utils.general as utils
    from training_pipeline import train

    # Applies optional utilities:
    # - disabling python warnings
    # - prints config
    utils.extras(hydra_config)

    # Train model
    train(hydra_config)


if __name__ == "__main__":
    main()
