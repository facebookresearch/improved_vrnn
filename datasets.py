#
# Copyright (c) Facebook, Inc. and its affiliates.
#
import numpy as np
import torch
import torch.utils.data.distributed
from torch.utils.data import DataLoader


def get_dataset(config):
    """Returns train/val dataloaders."""

    config['seq_len'] = config['n_ctx'] + config['n_steps']
    normalize = True

    if config['dataset'] == 'stochastic':
        from dataloaders.stochastic_mmnist import MovingMNIST

        train_dataset = MovingMNIST(
            True,
            seq_len=config['seq_len'],
            deterministic=False
        )

        val_dataset = MovingMNIST(
            True,
            seq_len=config['seq_len'],
            deterministic=False
        )

        img_ch = 1

    elif config['dataset'] == 'pushbair':
        from dataloaders.pushbair_loader import PushDataset

        train_dataset = PushDataset(
            'train',
            config['seq_len'],
            normalize=normalize,
        )

        val_dataset = PushDataset(
            'test',
            config['seq_len'],
            normalize=normalize,
        )

        img_ch = 3

    elif config['dataset'] == 'pushbair_fvd':
        from dataloaders.pushbair_fvd_loader import PushDataset

        train_dataset = PushDataset(
            'train',
            config['seq_len'],
            normalize=normalize,
        )

        val_dataset = PushDataset(
            'test',
            config['seq_len'],
            normalize=normalize,
        )

        img_ch = 3

    elif config['dataset'] == 'cityscapes':
        from dataloaders.cityscapes_loader import CityscapesDataset
        
        train_dataset = CityscapesDataset(
            'train_64',
            config['seq_len'],
            img_side=64,
            normalize=normalize,
            resize=False,
        )

        val_dataset = CityscapesDataset(
            'test_64',
            config['seq_len'],
            img_side=64,
            normalize=normalize,
            resize=False,
        )

        img_ch = 3

    elif config['dataset'] == 'cityscapes128':
        from dataloaders.cityscapes_loader import CityscapesDataset
        
        train_dataset = CityscapesDataset(
            'train_128',
            config['seq_len'],
            img_side=128,
            normalize=normalize,
            resize=False,
        )

        val_dataset = CityscapesDataset(
            'test_128',
            config['seq_len'],
            img_side=128,
            normalize=normalize,
            resize=False,
        )

        img_ch = 3


    def init_fun(worker_id):
        return np.random.seed()

    if config['multigpu']:
        train_sampler =  torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler =  torch.utils.data.distributed.DistributedSampler(val_dataset)

        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'],
            sampler=train_sampler,
            # shuffle=True,
            num_workers=config['n_workers'],
            worker_init_fn=init_fun
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'],
            sampler=val_sampler,
            # shuffle=False,
            num_workers=config['n_workers'],
            worker_init_fn=init_fun
        )
        
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['n_workers'],
            worker_init_fn=init_fun,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'],
            shuffle=True,
            # shuffle=False,
            num_workers=config['n_workers'],
            worker_init_fn=init_fun,
        )

    config['img_ch'] = img_ch
    config['batches_per_epoch'] = len(train_dataset)//config['batch_size']

    return train_loader, val_loader
