#
# Copyright (c) Facebook, Inc. and its affiliates.
#
import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import numpy as np
import losses


def prepare_batch(batch, config):
    """Prepare a batch to be fed to the model."""
    # Separate frames and actions
    frames, idxs = batch

    # Move frames to GPU
    frames = frames.to(config['device'])
    idxs = idxs.to(config['device'])

    return frames, idxs 


def sample_pass(frames, model, config, n_steps, use_mean=False, scale_var=1.):

    # Prepare frames
    input_frames = frames.clone().detach()

    preds = []
    for step_idx in range(n_steps):
        # print(step_idx)
        (cur_outs, cur_priors, cur_posteriors), _ = model(input_frames, config, True, use_mean=use_mean, scale_var=scale_var)
        preds.append(cur_outs[:, step_idx])
        input_frames[:, config['n_ctx'] + step_idx] = preds[-1]
    preds = torch.stack(preds, 1)

    return preds, cur_priors, cur_posteriors


def sample_step(model, config, outputs, n_steps=None, use_mean=False, scale_var=1.):
    """Get a sample (prior) from the model."""

    with torch.no_grad():
        inputs = outputs.clone().detach()

        if n_steps is None:
            n_steps = config['n_steps']
        else:
            b, t, c, h, w = inputs.shape
            missing_frames = n_steps - config['n_steps']
            aux = torch.zeros(b, missing_frames, c, h, w).to(inputs.device)
            inputs = torch.cat([inputs, aux], 1)
            outputs = torch.cat([outputs, aux], 1)

        inputs[:, config['n_ctx']:] = 0.
        (preds, _, _) = sample_pass(inputs, model, config, n_steps, use_mean=use_mean, scale_var=scale_var)
        targets = outputs
        ctx = outputs[:, :config['n_ctx']]

        # Prepare predictions
        preds = torch.cat([ctx, preds], 1)

    return (preds, targets), None


def reconstruction_step(model, config, inputs, use_mean=False):
    """Get a reconstruction (posterior) from the model."""
    outputs = inputs.clone().detach()

    with torch.no_grad():
        (preds, _, _), stored_vars = model(inputs, config, False)

        targets = outputs
        ctx = outputs[:, :config['n_ctx']]
        preds = torch.cat([ctx, preds], 1)

    return (preds, targets), stored_vars


def train_step(model, config, inputs, optimizer, batch_idx, logger=None):
    """Training step for the model."""

    outputs = inputs.clone().detach()

    # Forward pass
    (preds, priors, posteriors), stored_vars = model(inputs, config, False)

    # Accumulate preds and select targets
    targets = outputs[:, config['n_ctx']:]

    # Compute the reconstruction loss
    loss_rec = losses.reconstruction_loss(config, preds, targets)

    # Compute the prior loss
    if config['beta'] > 0:
        loss_prior = losses.kl_loss(config, priors, posteriors)
        loss = loss_rec + config['beta']*loss_prior
    else:
        loss_prior = 0.
        loss = loss_rec

    # Backward pass and optimizer step
    optimizer.zero_grad()
    if config['apex']:
        from apex import amp
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    optimizer.step()

    # Logs 
    if logger is not None:
        logger.scalar('train_loss_rec', loss_rec.item())
        logger.scalar('train_loss', loss.item())
        if config['beta'] > 0:
            logger.scalar('train_loss_prior', loss_prior.item())

    return preds, targets, priors, posteriors, loss_rec, loss_prior, loss, stored_vars


def test_step(model, config, inputs, logger=None):
    """Reconstruction Test step during training."""
    outputs = inputs.clone().detach()

    with torch.no_grad():
        (preds, priors, posteriors), stored_vars = model(
            inputs,
            config, 
            False,
        )

        # Accumulate preds and select targets
        targets = outputs[:, config['n_ctx']:]

        # Compute the reconstruction and prior loss
        loss_rec = losses.reconstruction_loss(config, preds, targets)
        if config['beta'] > 0:
            loss_prior = losses.kl_loss(config, priors, posteriors)
            loss = loss_rec + config['beta']*loss_prior
        else:
            loss = loss_rec

    # Logs 
    if logger is not None:
        logger.scalar('test_loss_rec', loss_rec.item())
        logger.scalar('test_loss', loss.item())
        if config['beta'] > 0:
            logger.scalar('test_loss_prior', loss_prior.item())


def train_print_status(log):
    """Print a line with information on the previous training iterations."""
    line = ''
    line += '[Epoch {}] '.format(log.epoch)
    line += 'Iteration {} '.format(log.abs_train_it)
    line += 'TRAIN '
    line += 'Rec: {:2.4f} '.format(log.last_value('train_loss_rec'))
    line += 'Prior: {:2.4f} '.format(log.last_value('train_loss_prior'))
    line += 'Loss: {:2.4f} '.format(log.last_value('train_loss'))

    log.print(line)


def test_print_status(log):
    line = ''
    line += '[Epoch {}] '.format(log.epoch)
    line += 'Iteration {} '.format(log.abs_train_it)
    line += 'TEST ' 
    line += 'Rec: {:2.4f} '.format(log.last_value('test_loss_rec'))
    line += 'Prior: {:2.4f} '.format(log.last_value('test_loss_prior'))
    line += 'Loss: {:2.4f} '.format(log.last_value('test_loss'))

    log.print(line)


def reconstruction_step_with_zs(model, config, inputs, use_mean=False):
    """Get a reconstruction (posterior) from the model."""
    outputs = inputs.clone().detach()

    with torch.no_grad():
        (preds, p, q), stored_vars = model(inputs, config, False)

        targets = outputs
        ctx = outputs[:, :config['n_ctx']]
        preds = torch.cat([ctx, preds], 1)

    return p, q


def sample_step_with_zs(model, config, outputs, n_steps=None, use_mean=False):

    with torch.no_grad():
        inputs = outputs.clone().detach()

        if n_steps is None:
            n_steps = config['n_steps']
        else:
            b, t, c, h, w = inputs.shape
            missing_frames = n_steps - config['n_steps']
            aux = torch.zeros(b, missing_frames, c, h, w).to(inputs.device)
            inputs = torch.cat([inputs, aux], 1)
            outputs = torch.cat([outputs, aux], 1)

        inputs[:, config['n_ctx']:] = 0.
        (preds, priors, posteriors) = sample_pass(inputs, model, config, n_steps, use_mean=use_mean)
        targets = outputs
        ctx = outputs[:, :config['n_ctx']]

        # Prepare predictions
        preds = torch.cat([ctx, preds], 1)

    return (preds, targets, priors, posteriors), None
