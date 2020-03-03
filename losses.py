#
# Copyright (c) Facebook, Inc. and its affiliates.
#
import numpy as np
import torch
import torch.nn.functional as F

from flows import gaussian_diag_logps


def kl_loss(config, priors, posteriors):
    """Prior loss for a VAE."""
    all_kl = []
    for p, q in zip(priors, posteriors):
        dist_kl = [] 
        p_m = p[0]
        p_lv = p[1]
        q_m = q[0] 
        q_lv = q[1]
        q_z0 = q[2]
        q_zk = q[3]
        q_ladj = q[4]

        for t_idx in range(q_z0.shape[1]):
            cur_z0 = q_z0[:, t_idx]
            cur_zk = q_zk[:, t_idx]
            cur_p_means = p_m[:, t_idx]
            cur_p_logvar = p_lv[:, t_idx]
            cur_q_means = q_m[:, t_idx]
            cur_q_logvar = q_lv[:, t_idx]
            log_p = gaussian_diag_logps(cur_p_means, cur_p_logvar, sample=cur_zk)
            log_q = gaussian_diag_logps(cur_q_means, cur_q_logvar, sample=cur_z0)
            if q_ladj is not None:
                cur_ladj = q_ladj[:, t_idx]
                cur_kl = (log_q - log_p).sum(-1).sum(-1).sum(-1) - (cur_ladj).sum(-1).sum(-1).sum(-1)
            else:
                cur_kl = (log_q - log_p).sum(-1).sum(-1).sum(-1)

            dist_kl.append(cur_kl)
        dist_kl = torch.stack(dist_kl, 1).sum(1)
        all_kl.append(dist_kl)
    all_kl = torch.stack(all_kl, 1).sum(1).mean()
    loss_kl = all_kl

    return loss_kl


def reconstruction_loss(config, preds, targets):
    """Returns the reconstruction loss of the VAE.
    
    Mean over batch, sum over the rest of dimensions (including time).

    Args:
        config:
        preds:
        targets:
    """
    if config['rec_loss'] == 'l2':
        loss = F.mse_loss(preds, targets, reduction='none')
        loss = loss.sum(-1).sum(-1).sum(-1).sum(-1).mean()

    elif config['rec_loss'] == 'l1':
        loss = F.l1_loss(preds, targets, reduction='none')
        loss = loss.sum(-1).sum(-1).sum(-1).sum(-1).mean()

    elif config['rec_loss'] == 'bce':
        loss = F.binary_cross_entropy(preds, targets, reduction='none')
        while len(loss.shape) > 1:
            loss = loss.sum(-1)
        loss = loss.mean()

    return loss
