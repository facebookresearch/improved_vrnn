#
# Copyright (c) Facebook, Inc. and its affiliates.
#
"""
    init_model.py
    ~~~~~~~~~~~~~

    Model inits.
"""
import torch


def init_model(config):

    # Return model by name
    if config['model'] == 'vrnn':
        from models.vrnn import Model
        model = Model(
            config['img_ch'],
            config['n_ctx'],
            n_z=config['n_z'] if 'n_z' in config else 10,
        ).to(config['device'])

    elif config['model'] == 'vrnn_hier':
        from models.vrnn_hier import Model
        model = Model(
            config['img_ch'],
            config['n_ctx'],
            n_z=config['n_z'] if 'n_z' in config else 10,
        ).to(config['device'])


    # Reload checkpoint if needed
    if config['checkpoint'] is not None:
        state_dict = torch.load(config['checkpoint'])

        aux_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module'):
                aux_state_dict[k[7:]] = v
            else:
                aux_state_dict[k] = v
        model.load_state_dict(aux_state_dict)

    return model
