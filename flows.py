#
# Copyright (c) Facebook, Inc. and its affiliates.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def gaussian_rsample(mean, logvar, use_mean=False):
    if use_mean:
        return mean
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mean)


def gaussian_diag_logps(mean, logvar, sample):
    const = 2 * np.pi * torch.ones_like(mean).to(mean.device)
    return -0.5 * (torch.log(const) + logvar + (sample - mean)**2 / torch.exp(logvar))
