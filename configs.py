#
# Copyright (c) Facebook, Inc. and its affiliates.
#
"""
    configs.py
    ~~~~~~~~~~

    Utilities to load and save model configurations.
"""
import os
import json


def args2config(args):
    return vars(args)

def save_config(out_path, config):
    with open(out_path, 'w') as f:
        json.dump(config, f, indent=2)

def load_config(path):
    with open(path, 'r') as f:
        config = json.load(f)

    return config

def join_configs(config1, config2):
    return {**config1, **config2}

