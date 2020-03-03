#
# Copyright (c) Facebook, Inc. and its affiliates.
#
import os
import time
import random
import argparse
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.distributions import kl_divergence, Normal, MultivariateNormal
from tqdm import tqdm

import train_fns
import utils
import configs
from utils import torch2img, save_gif
from logger import MyLogger
from datasets import get_dataset
from init_model import init_model


torch.backends.cudnn.benchmark = True

# np.random.seed(1234)
# torch.manual_seed(1234)


def parse_args():
    """Returns a configuration from command line arguments."""

    # Main options
    parser = argparse.ArgumentParser()

    parser.add_argument('--out_dir', required=True, type=str,
                        help='Output directory')
    parser.add_argument('--exp_name', default=None, type=str,
                        help='Name for the experiment output folder')
    parser.add_argument('--dataset', required=True, type=str,
                        help='Dataset to use')

    # Model options
    parser.add_argument('--model', required=True, type=str,
                        help='Type of Encoder/Decoder to use')
    parser.add_argument('--rec_loss', default='l1', type=str,
                        choices=['l1', 'l2', 'bce'],
                        help='Reconstruction loss to use')
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='Resume training with this checkpoint path')

    # Hyperparameters
    parser.add_argument('--n_ctx', required=True, type=int,
                        help='Number of context frames to use')
    parser.add_argument('--n_steps', required=True, type=int,
                        help='Number of steps to unroll the model for')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate for the optimizer')
    parser.add_argument('--n_z', default=10, type=int,
                        help='Number of latents to use.')
    parser.add_argument('--beta', default=1, type=float,
                        help='Weight for the KL loss in the VAE objective')
    parser.add_argument('--beta_wu', default=0, type=int,
                        help='Use a warm-up schedule for the beta parameter')

    # Experiment options
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Number of examples per batch')
    parser.add_argument('--log_freq', default=100, type=int,
                        help='Display information every X batches')
    parser.add_argument('--sample_freq', default=1, type=int,
                        help='Log samples every X epochs')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='Save model every X epochs')
    parser.add_argument('--n_workers', default=4, type=int,
                        help='Number of data loading threads')
    parser.add_argument('--gpu', default=1, type=int,
                        help='Use GPU')
    parser.add_argument('--max_epochs', default=500, type=int,
                        help='Maximum number of epochs')
    parser.add_argument('--test_batches', default=10, type=int,
                        help='Number of batches to test the model for.')

    # Distributed options and float16 options
    parser.add_argument('--apex', action='store_true', default=False,
                        help='Use APEX (float16)')
    parser.add_argument('--multigpu', default=0, type=int,
                        help='Use multi GPU training')
    parser.add_argument('--dist-url', default=None, type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')

    args = parser.parse_args()

    return vars(args)


def main(config):

    # Set up MultiGPU training
    if config['multigpu']:
        rank = int(os.environ.get("SLURM_PROCID"))
        n_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES"))
        world_size = int(os.environ.get("SLURM_NTASKS_PER_NODE"))*n_nodes
        local_rank = int(os.environ.get("SLURM_LOCALID"))
        print('LOCAL RANK {}'.format(local_rank))

        dist.init_process_group(
            backend=config['dist_backend'],
            init_method=config['dist_url'],
            world_size=world_size,
            rank=rank
        )

        config['rank'] = rank
        config['local_rank'] = local_rank

    # Load configuration and set up experiment
    if not config['multigpu'] or rank == 0:
        # config['out_dir'] = os.path.join(config['out_dir'], config['exp_name'])
        log_dir = os.path.join(config['out_dir'], 'logs')
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        log = MyLogger(log_dir)
        config_path = os.path.join(config['out_dir'], 'config.json')
        os.makedirs(config['out_dir'], exist_ok=True)
        os.makedirs(os.path.join(config['out_dir'], 'checkpoints'), exist_ok=True)
        print('Output directory prepared')
    else:
        log = None

    # Set device for the model
    if config['multigpu']:
        config['device'] = 'cuda:{}'.format(local_rank)
    else:
        config['device'] = 'cuda' if config['gpu'] else 'cpu'

    # Load dataset
    train_loader, val_loader = get_dataset(config)
    if not config['multigpu'] or rank == 0:
        print('Dataset loaded')

    # Load model
    model = init_model(config)
    if not config['multigpu'] or rank == 0:
        print(model)
        print('Model loaded')

    # Load optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=1e-4,
    )
    if not config['multigpu'] or rank == 0:
        print('Optimizer initialized')


    # Set up apex and distributed training
    if config['apex']:
        from apex import amp, parallel
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        if config['multigpu']:
            with torch.cuda.device(config['local_rank']):
                model = parallel.DistributedDataParallel(model)
            
    else:
        if config['multigpu']:
            model = torch.nn.parallel.DistributedDataParallel(
                model, 
                device_ids=[config['local_rank']], 
                output_device=config['local_rank'],
            )

    # Save updated config
    if not config['multigpu'] or rank == 0:
        configs.save_config(config_path, config)

    # TODO: move this to train functions
    def generate_samples(frames, split):
        all_preds = []
        (preds, targets), _ = train_fns.sample_step(model, config, frames, use_mean=True)
        all_preds.append(preds)
        for idx in range(5):
            (preds, targets), _ = train_fns.sample_step(model, config, frames)
            all_preds.append(preds)
        preds = torch.cat(all_preds, -1)
        video = torch.cat([targets, preds], -1)
        log.video('{}_sample'.format(split), video)

        (preds, targets), _ = train_fns.reconstruction_step(model, config, frames)
        video = torch.cat([targets, preds], -1)
        log.video('{}_reconstruction'.format(split), video)


    # Main loop
    abs_batch_idx = 1
    beta2 = config['beta']
    for epoch_idx in range(config['max_epochs']):

        t1 = time.time()

        # Train iterations
        model.train()
        for batch_idx, train_batch in enumerate(train_loader):

            # if batch_idx == 1:
            #     break

            # Change learning rate
            if config['multigpu']:

                warmup_iters = config['batches_per_epoch']*5/world_size
                lr1 = config['lr']
                lr2 = config['lr']*config['batch_size']*world_size/16
                warmup_step = (lr2 - lr1) / warmup_iters

                if abs_batch_idx < warmup_iters:
                    # Gradual scaling
                    lr = abs_batch_idx*warmup_step + lr1
                else:
                    # Decay learning rate
                    lr = lr2

            else:
                lr = config['lr']

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Warmup beta
            if config['beta_wu']:
                if not config['multigpu']:
                    world_size = 1
                beta1 = 0
                bn_epochs = 20
                bwarmup_iters = config['batches_per_epoch']*bn_epochs/world_size
                bwarmup_step = (beta2 - beta1)/bwarmup_iters

                if abs_batch_idx < bwarmup_iters:
                    config['beta'] = abs_batch_idx*bwarmup_step + beta1
                else:
                    config['beta'] = beta2

            t2 = time.time()

            # if batch_idx == 1: break
            frames, example_idxs = train_fns.prepare_batch(train_batch, config)
            train_fns.train_step(
                model,
                config, 
                frames, 
                optimizer,
                batch_idx,
                log
            )

            t3 = time.time()

            # print('TIME {:.4f}/{:.4f}'.format(t2 -t1, t3 - t2))
            t1 = time.time()

            abs_batch_idx += 1

            if not config['multigpu'] or rank == 0:
                log.increase_train()

                if (batch_idx + 1) % config['log_freq'] == 0:
                    log.dump_scalars('train')
                    train_fns.train_print_status(log)

        if not config['multigpu'] or rank == 0:
            log.dump_scalars('train')
            train_fns.train_print_status(log)

        # Validation iterations
        model.eval()
        for batch_idx, val_batch in enumerate(val_loader):

            if batch_idx == config['test_batches']:
                break

            frames, example_idxs = train_fns.prepare_batch(val_batch, config)

            train_fns.test_step(
                model,
                config, 
                frames,
                log
            )

            if not config['multigpu'] or rank == 0:
                log.increase_test()

        # Bookkeeping
        if not config['multigpu'] or rank == 0:

            # Train reconstruction and samples
            frames, example_idxs = train_fns.prepare_batch(train_batch, config)
            generate_samples(frames, 'train')
            log.print('Train samples saved')

            # Test reconstructions and samples
            frames, example_idxs = train_fns.prepare_batch(val_batch, config)
            generate_samples(frames, 'val')
            log.print('Test samples saved')

            # Print information and increase epoch
            log.dump_scalars('test')
            train_fns.test_print_status(log)
            log.increase_epoch()

            # Save model 
            if (epoch_idx + 1) % config['save_freq'] == 0:
                torch.save(model.state_dict(), os.path.join(config['out_dir'], 'checkpoints', '{:0>5d}.pth'.format(epoch_idx + 1)))
                log.print('Model saved')


if __name__ == '__main__':
    config = parse_args()
    main(config)
