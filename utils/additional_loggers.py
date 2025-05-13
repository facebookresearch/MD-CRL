# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import wandb


def get_img_rec_table_data(imgs, step, num_samples_to_log):
    
    imgs = [wandb.Image(x) for x in imgs[:num_samples_to_log]]

    columns = ["step", "x"]
    data = [
        [step, x]
        for x in imgs
    ]
    return columns, data


def add_column_to_table_data(columns, data, new_col_name, new_col_data):
    # TODO: Verify correctness
    columns.append(new_col_name)
    for row, new_cell_value in zip(data, new_col_data):
        row.append(new_cell_value)