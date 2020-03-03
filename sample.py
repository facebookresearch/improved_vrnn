#
# Copyright (c) Facebook, Inc. and its affiliates.
#
import os
import time
import random
import argparse
import skimage.io
from datetime import datetime
from tqdm import tqdm
import socket
import shutil

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.distributions import kl_divergence, Normal, MultivariateNormal
from joblib import Parallel, delayed

import configs
import train_fns 
from init_model import init_model
from utils import torch2img, save_gif
from datasets import get_dataset


# Initial setup
torch.backends.cudnn.benchmark = True
np.random.seed(1234)
torch.manual_seed(1234)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', required=True, type=str,
                        help='Path to checkpoint of the model')
    parser.add_argument('--n_steps', default=None, type=int,
                        help='Number of steps to predict')
    parser.add_argument('--n_seqs', default=100, type=int,
                        help='Number of sequences/examples to predict')
    parser.add_argument('--n_samples', default=10, type=int,
                        help='Number of different samples per sequence to generate')
    args = parser.parse_args()
    config = vars(args)

    return config


def save_sample_png(sample_dir, frame, frame_id):
    out_path = os.path.join(sample_dir, '{:0>4}.png'.format(frame_id))
    frame = frame*255.
    if frame.shape[2] > 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        frame = frame[:, :, 0]
    cv2.imwrite(out_path, frame)


def main(config):

    # Load checkpoint config
    old_config = config
    config_dir = os.path.dirname(os.path.dirname(config['checkpoint']))
    config_path = os.path.join(config_dir, 'config.json')
    config = configs.load_config(config_path)

    # Remove multigpu flags and adjust batch size
    config['multigpu'] = 0
    config['batch_size'] = 1

    # Overwrite config params
    config['checkpoint'] = old_config['checkpoint']
    if old_config['n_steps'] is not None:
        config['n_steps'] = old_config['n_steps']
    config['n_seqs'] = old_config['n_seqs']
    config['n_samples'] = old_config['n_samples']

    # Set up device
    local_rank = 0
    config['local_rank'] = 0
    config['device'] = 'cuda:{}'.format(local_rank)

    train_loader, val_loader = get_dataset(config)
    print('Dataset loaded')

    model = init_model(config)
    print(model)
    print('Model loaded')

    # Define output dirs
    out_dir = config_dir
    samples_dir = os.path.join(out_dir, 'samples')
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir, exist_ok=True)

    # Define saving function
    def save_samples(preds, gt, ctx, out_dir, seq_id):

        # Compute number of samples and sequences
        seq_dir = os.path.join(samples_dir, '{:0>4}'.format(seq_id))
        n_samples = len(preds)
        timesteps = gt.shape[1]

        # Save samples
        for sample_id in range(n_samples):
            sample_dir = os.path.join(seq_dir, '{:0>4}'.format(sample_id))
            os.makedirs(sample_dir, exist_ok=True)
            Parallel(n_jobs=20)(delayed(save_sample_png)(sample_dir, frame, f_id) for f_id, frame in enumerate(preds[sample_id]))

        # Save ctx
        sample_dir = os.path.join(seq_dir, 'ctx')
        os.makedirs(sample_dir, exist_ok=True)
        Parallel(n_jobs=20)(delayed(save_sample_png)(sample_dir, frame, f_id) for f_id, frame in enumerate(ctx[0]))

        # Save gt
        sample_dir = os.path.join(seq_dir, 'gt')
        os.makedirs(sample_dir, exist_ok=True)
        Parallel(n_jobs=20)(delayed(save_sample_png)(sample_dir, frame, f_id) for f_id, frame in enumerate(gt[0]))

    model.eval()
    n_seqs = 0
    # for batch_idx, batch in enumerate(tqdm(val_loader, desc='Sequence loop')):
    for batch_idx, batch in enumerate(val_loader):

        if n_seqs >= config['n_seqs']: 
            break

        frames, idxs = train_fns.prepare_batch(batch, config)

        # Find id of the sequence and decide whether to work on it or not
        sequence_id = idxs[0]
        sequence_dir = os.path.join(samples_dir, '{:0>4}'.format(sequence_id))
        if os.path.exists(sequence_dir):
            n_seqs += frames.shape[0]
            continue
        os.makedirs(sequence_dir, exist_ok=True)

        batch_size = 1
        frames = frames.repeat(batch_size, 1, 1, 1, 1)
        samples_done = 0
        all_preds = []

        sampling_ok = True
        while samples_done < config['n_samples']:
            try:
                (preds, targets), _ = train_fns.sample_step(model, config, frames)
            except:
                sampling_ok = False
                break

            preds = preds[:, config['n_ctx']:].contiguous()
            preds = preds.detach()
            targets = targets.detach()
            all_preds.append(preds)
            samples_done += batch_size

        if not sampling_ok:
            continue

        # Trim extra samples
        all_preds = torch.cat(all_preds, 0)
        all_preds = all_preds[:config['n_samples']]

        # Convert to numpy
        ctx = targets[:, :config['n_ctx']]
        targets = targets[:, config['n_ctx']:]
        targets = targets.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)
        ctx = ctx.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)
        all_preds = all_preds.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)

        # Save samples to PNG files
        save_samples(all_preds, targets, ctx, out_dir, sequence_id)

        # Update number of samples
        n_seqs += frames.shape[0]

    print('All done')



if __name__ == '__main__':
    config = parse_args()
    main(config)
