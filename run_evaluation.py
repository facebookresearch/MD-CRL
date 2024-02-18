# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from slot_based_disentanglement.utils import hydra_custom_resolvers
import hydra
from omegaconf import DictConfig



@hydra.main(config_path="configs", config_name="evaluate_root")
def main(hydra_config: DictConfig):
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934

    import slot_based_disentanglement.utils.general as utils
    from slot_based_disentanglement.evaluation_pipeline import evaluate

    # Applies optional utilities:
    # - disabling python warnings
    # - prints config
    utils.extras(hydra_config)

    # Evaluate model
    evaluate(hydra_config)


if __name__ == "__main__":
    main()