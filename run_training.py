# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from utils import hydra_custom_resolvers
import hydra
from omegaconf import DictConfig



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
