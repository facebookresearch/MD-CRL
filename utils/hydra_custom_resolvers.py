from omegaconf import OmegaConf
import math
from functools import reduce
import operator
import os
from pathlib import Path


def add_args(*args):
    return sum(float(x) for x in args)


def add_args_int(*args):
    return int(sum(float(x) for x in args))


def multiply_args(*args):
    return int(reduce(operator.mul, (float(x) for x in args), 1))
    # return math.prod(float(x) for x in args)


def multiply_args_int(*args):
    return int(reduce(operator.mul, (float(x) for x in args), 1))
#     return int(math.prod(float(x) for x in args))

def concat_str_args(*str_list):
    if isinstance(str_list, tuple): # handling some special cases
        str_list = str_list[0]
    return ''.join(str_list)

def resolve_tuple(*args):
    return(tuple(args))

def floor_division(dividend, divisor):
    return dividend // divisor

def run_name_ckpt_path(ckpt_path):
    return os.path.basename(Path(ckpt_path))

def best_ckpt_path_retrieve(path, ckpt_params):
    isExist = os.path.exists(path)
    if not isExist:
        raise Exception(f"Path `{path}` does not exist!")

    isDir = os.path.isdir(path)
    if not isDir:
        raise Exception(f"Path `{path}` does not correspond to a directory!")

    dir_list = os.listdir(path)
    target_dir_id = [idx for idx, directory in enumerate(dir_list) if f"{ckpt_params['param_name']}={ckpt_params['param_value']}" in directory][0]
    
    best_ckpt_file_path = os.path.join(path, dir_list[target_dir_id], "best_ckpt_path.txt")
    best_ckpt_exists = os.path.isfile(best_ckpt_file_path)
    if not best_ckpt_exists:
        raise Exception(f"File `{best_ckpt_file_path}` does not exist. trainer.fit() might have crashed before saving the path to the best ckpt!")

    with open(best_ckpt_file_path, "r") as f:
        best_ckpt_name = f.readlines()[0]

    return os.path.join(path, dir_list[target_dir_id], "checkpoints", best_ckpt_name)


def module_freeze_resolve(encoder_freeze, slot_attention_freeze_only, attention_key_freeze_only):
    if attention_key_freeze_only:
        return "attention"
    elif slot_attention_freeze_only:
        return "slot_attention"
    elif encoder_freeze:
        return "encoder-decoder"
    else:
        print("No part of the encoder-decoder is frozen and all will be trained.")

import torch
import string
import numpy as np
def retrieve_encoder_state_dict_from_full_ckpt(path):
    if path is None:
        return
    # note that this function runs every time the hydra config is resolved, therefore all instances of state_dict that are saved should be deleted
    # accordingly.
    model = torch.load(path)
    state_dict = model["state_dict"]
    keys = state_dict.copy().keys()
    for key in keys:
        state_dict[key.replace("model.", "")] = state_dict.pop(key)
    # save path should be unique because even though it is deleted soon, we don't want to have any clashes when more than
    # one instance of the job are running
    letters = string.ascii_lowercase # all lowercase letters of the alphabet
    save_path_appendix = ''.join(np.random.choice(list(letters)) for i in range(6))
    print(os.getcwd())
    torch.save(state_dict, f"state_dict_{save_path_appendix}.pth")

    return f"state_dict_{save_path_appendix}.pth"

def retrieve_num_domains(path):
    if path is None:
        return
    # inside the path directory, there should be a file called config_tree.txt
    # this file contains the config tree of the job that was run
    # we can use this to retrieve the number of domains the pattern is like
    # num_domains: 8
    with open(os.path.join(path, "config_tree.txt"), "r") as f:
        config_tree = f.readlines()
    num_domains = [line.split(":")[1].strip() for line in config_tree if "num_domains" in line][0]
    return int(num_domains)

def retrieve_x_dim(path):
    # x_dim is the dimension of the encoded image which will be
    # called z_dim by the mlp autoencoder
    if path is None:
        return
    # inside the path directory, there should be a file called config_tree.txt
    # this file contains the config tree of the job that was run
    # we can use this to retrieve the number of domains the pattern is like
    # z_dim: 256
    with open(os.path.join(path, "config_tree.txt"), "r") as f:
        config_tree = f.readlines()
    x_dim = [line.split(":")[1].strip() for line in config_tree if "z_dim" in line][0]
    return int(x_dim)

OmegaConf.register_new_resolver("add", add_args)
OmegaConf.register_new_resolver("mult", multiply_args)

OmegaConf.register_new_resolver("add_int", add_args_int)
OmegaConf.register_new_resolver("mult_int", multiply_args_int)

OmegaConf.register_new_resolver("join_str", concat_str_args)

OmegaConf.register_new_resolver("tuple", resolve_tuple)

OmegaConf.register_new_resolver("floor_div", floor_division)

OmegaConf.register_new_resolver("run_name_ckpt", run_name_ckpt_path)

OmegaConf.register_new_resolver("path_to_best_ckpt", best_ckpt_path_retrieve)

OmegaConf.register_new_resolver("retrieve_encoder_state_dict", retrieve_encoder_state_dict_from_full_ckpt)

OmegaConf.register_new_resolver("retrieve_num_domain", retrieve_num_domains)

OmegaConf.register_new_resolver("retrieve_x_dimension", retrieve_x_dim)
