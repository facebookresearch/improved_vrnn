#
# Copyright (c) Facebook, Inc. and its affiliates.
#
import os
import sys
import logging
import shutil
from datetime import datetime

import numpy as np

import utils


def get_logger(filepath=None):
    """Return a console + file logger."""
    log = logging.getLogger('Video')

    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    log_format = logging.Formatter(
        fmt='[%(asctime)s][%(filename)s:%(lineno)d] %(message)s'
    )

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(log_format)
    log.addHandler(console_handler)

    if filepath is not None:
        file_handler = logging.FileHandler(filepath, mode='a')
        file_handler.setFormatter(log_format)
        log.addHandler(file_handler)

    log.setLevel(logging.INFO)

    return log


class Counter():
    def __init__(self, name):
        self.name = name
        self.samples = []
    
    def add(self, x):
        self.samples.append(x)

    def mean(self):
        mean = np.mean(self.samples)
        self.samples = []

        return mean

    def last(self):
        last = self.samples[-1]
        self.samples = []

        return last


class MyLogger():
    """Custom logging class."""

    def __init__(self, out_dir, epoch=0, train_it=0, test_it=0):
        self.out_dir = out_dir
        log_path = os.path.join(out_dir, 'log.txt')
        self.log = get_logger(log_path)
        self.epoch = epoch
        self.abs_train_it = train_it
        self.abs_test_it = test_it
        self.rel_train_it = 0
        self.rel_test_it = 0
        self.counters = {}

    def print(self, msg):
        self.log.info(msg)

    def scalar(self, name, value):
        if name not in self.counters:
            self.counters[name] = Counter(name)

        self.counters[name].add(value)

    def video(self, name, video):
        """Save a BxTxCxHxW video into a gif."""
        gif_dir = os.path.join(self.out_dir, 'gifs')
        os.makedirs(gif_dir, exist_ok=True)
        fname = '{}_{}_{}.gif'.format(
            name,
            self.epoch,
            self.abs_train_it,
        )
        fname = os.path.join(gif_dir, fname)
        utils.save_gif(fname, video)

    def increase_train(self):
        self.abs_train_it += 1
        self.rel_train_it += 1

    def increase_test(self):
        self.abs_test_it += 1
        self.rel_test_it += 1

    def increase_epoch(self):
        self.epoch += 1
        self.rel_train_it = 0
        self.rel_test_it = 0


    def dump_scalars(self, split=None):
        scalars_dir = os.path.join(self.out_dir, 'scalars')
        os.makedirs(scalars_dir, exist_ok=True)

        for k, counter in self.counters.items():
            if split is not None:
                if not counter.name.startswith(split):
                    continue
            counter_path = os.path.join(scalars_dir, '{}.txt'.format(counter.name))

            with open(counter_path, 'a') as f: 
                f.write('{},{:2.4f}\n'.format(
                    self.abs_train_it,
                    counter.mean()
                ))


    def last_value(self, name):

        scalars_dir = os.path.join(self.out_dir, 'scalars')
        counter_path = os.path.join(scalars_dir, '{}.txt'.format(name))

        with open(counter_path, 'r') as f:
            last_line = f.readlines()[-1].strip().split(',')
            last_value = float(last_line[1])

        return last_value

