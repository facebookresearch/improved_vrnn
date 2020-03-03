#
# Copyright (c) Facebook, Inc. and its affiliates.
#
"""
    s2s_convlstm.py
    ~~~~~~~~~~~

    Seq2Seq LSTM with ConvLSTM.
"""
# Imports ---------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from functools import partial

import layers
import flows
# ------------------------------------------------------------------------------


class Model(nn.Module):

    def __init__(self, img_ch, n_ctx,
            n_hid=64, 
            n_z=10,
            enc_dim=512, 
            share_prior_enc=False, 
            reverse_post=False,
        ):
        super().__init__()

        self.n_ctx = n_ctx
        self.enc_dim = enc_dim

        self.latent_emb_net = nn.ModuleList([
            layers.DcConv(img_ch, n_hid, 4, 2, 1),
            layers.DcConv(n_hid, n_hid*2, 4, 2, 1),
            layers.DcConv(n_hid*2, n_hid*4, 4, 2, 1),
            layers.DcConv(n_hid*4, n_hid*8, 4, 2, 1),
            layers.DcConv(n_hid*8, enc_dim*2, 4, 1, 0, norm=partial(nn.GroupNorm, 1)),
        ])

        self.det_emb_net = nn.ModuleList([
            layers.DcConv(img_ch, n_hid, 4, 2, 1),
            layers.DcConv(n_hid, n_hid*2, 4, 2, 1),
            layers.DcConv(n_hid*2, n_hid*4, 4, 2, 1),
            layers.DcConv(n_hid*4, n_hid*8, 4, 2, 1),
            layers.DcConv(n_hid*8, enc_dim*2, 4, 1, 0, norm=partial(nn.GroupNorm, 1)),
        ])

        mult = 1
        self.render_net = nn.ModuleList([
            layers.DcUpConv(enc_dim, n_hid*8, 4, 1, 0),
            layers.ConvLSTM(n_hid*8, n_hid*8),
            layers.DcUpConv(n_hid*8*mult, n_hid*4, 4, 2, 1),
            layers.ConvLSTM(n_hid*4, n_hid*4),
            layers.DcUpConv(n_hid*4*mult, n_hid*2, 4, 2, 1),
            layers.ConvLSTM(n_hid*2, n_hid*2),
            layers.DcUpConv(n_hid*2*mult, n_hid, 4, 2, 1),
            layers.ConvLSTM(n_hid, n_hid),
            layers.DcUpConv(n_hid*mult, n_hid, 4, 2, 1),
            layers.TemporalConv2d(n_hid*1, img_ch, 3, 1, 1),
        ])

        self.det_init_net = nn.Sequential(
            layers.DcConv(2*enc_dim*self.n_ctx, 2*enc_dim*self.n_ctx, 1),
            layers.TemporalConv2d(2*enc_dim*self.n_ctx, 2*enc_dim, 1)
        )

        self.prior_net = nn.ModuleList([
            layers.TemporalConv2d(enc_dim*2, n_z*2, 1),
            layers.ConvLSTM(n_z*2, enc_dim, norm=True),
            layers.TemporalConv2d(enc_dim, n_z*2, 1),
        ])

        self.posterior_net = nn.ModuleList([
            layers.TemporalConv2d(enc_dim*2, n_z*2, 1),
            layers.ConvLSTM(n_z*2, enc_dim, norm=True),
            layers.TemporalConv2d(enc_dim, n_z*2, 1),
        ])

        self.forward_model = layers.ConvLSTM(n_z + enc_dim*2, enc_dim)

        self.prior_init_net = nn.Sequential(
            layers.DcConv(2*enc_dim*self.n_ctx, 2*enc_dim*self.n_ctx, 1),
            layers.TemporalConv2d(2*enc_dim*self.n_ctx, 2*enc_dim, 1)
        )

        self.posterior_init_net = nn.Sequential(
            layers.DcConv(2*enc_dim*self.n_ctx, 2*enc_dim*self.n_ctx, 1),
            layers.TemporalConv2d(2*enc_dim*self.n_ctx, 2*enc_dim, 1)
        )

        # Connection list
        self.det_init_connections = {
            1: 3,
            3: 2,
            5: 1,
            7: 0,
        }

        # Connection branches
        self.det_init_nets = nn.ModuleDict({
            'layer_3': nn.Sequential(
                layers.DcConv(n_hid*8*self.n_ctx, n_hid*8*self.n_ctx, 1),
                layers.TemporalConv2d(n_hid*8*self.n_ctx, n_hid*8*2, 1)
            ),
            'layer_2': nn.Sequential(
                layers.DcConv(n_hid*4*self.n_ctx, n_hid*4*self.n_ctx, 1),
                layers.TemporalConv2d(n_hid*4*self.n_ctx, n_hid*4*2, 1)
            ),
            'layer_1': nn.Sequential(
                layers.DcConv(n_hid*2*self.n_ctx, n_hid*2*self.n_ctx, 1),
                layers.TemporalConv2d(n_hid*2*self.n_ctx, n_hid*2*2, 1)
            ),
            'layer_0': nn.Sequential(
                layers.DcConv(n_hid*1*self.n_ctx, n_hid*1*self.n_ctx, 1),
                layers.TemporalConv2d(n_hid*1*self.n_ctx, n_hid*1*2, 1)
            ),
        })


    def det_emb(self, x):
        out = x
        skips = []
        for layer_idx, layer in enumerate(self.det_emb_net):
            out = layer(out)
            skips.append(out)

        return skips

    def latent_emb(self, x):
        out = x
        for layer_idx, layer in enumerate(self.latent_emb_net):
            out = layer(out)

        return out

    def prior(self, x, ctx, use_mean=False):

        dists = []

        out = x
        for layer_idx, layer in enumerate(self.prior_net):

            if isinstance(layer, layers.ConvLSTM):
                # Process Condition
                cur_ctx = ctx.view(ctx.shape[0], -1, ctx.shape[-2], ctx.shape[-1]).unsqueeze(1)
                cur_ctx = self.prior_init_net(cur_ctx)
                cur_ctx = cur_ctx.squeeze(1)

                # Run LSTM with the inputs from the previous layer
                out = layer(out, torch.chunk(cur_ctx, 2, 1))

            else:
                out = layer(out)

        # Tanh the activations
        out = torch.tanh(out)

        # Compute distribution stats
        mean, logvar = torch.chunk(out, 2, 2)

        # Generate sample from this distribution
        z0 = flows.gaussian_rsample(mean, logvar, use_mean=use_mean)

        dists.append([mean, logvar, z0, z0, None])

        return dists


    def posterior(self, x, ctx, use_mean=False):
        """
        Encode the posterior.
        """
        dists = []

        out = x
        for layer_idx, layer in enumerate(self.posterior_net):

            if isinstance(layer, layers.ConvLSTM):
                # Process Condition
                cur_ctx = ctx.view(ctx.shape[0], -1, ctx.shape[-2], ctx.shape[-1]).unsqueeze(1)
                cur_ctx = self.posterior_init_net(cur_ctx)
                cur_ctx = cur_ctx.squeeze(1)

                # Run LSTM with the inputs from the previous layer
                out = layer(out, torch.chunk(cur_ctx, 2, 1))

            else:
                out = layer(out)

        # Tanh the activations
        out = torch.tanh(out)

        # Compute distribution stats
        mean, logvar = torch.chunk(out, 2, 2)

        # Generate sample from this distribution
        z0 = flows.gaussian_rsample(mean, logvar, use_mean=use_mean)

        dists.append([mean, logvar, z0, z0, None])

        return dists

    def render(self, x, zs, ctx):
        """
        Generate frames.
        """
        b, t, c, h, w = x.shape

        # Process first context
        ctx_in = ctx[-1][:, :self.n_ctx].contiguous()
        ctx_in = ctx_in.view(b, -1, ctx_in.shape[-2], ctx_in.shape[-1]).unsqueeze(1)
        cond = self.det_init_net(ctx_in).squeeze(1)
        forward_in = torch.cat([x, zs[0]], 2)
        out = self.forward_model(forward_in, torch.chunk(cond, 2, 1))

        for layer_idx, layer in enumerate(self.render_net):

            # print(layer_idx, '->', out.shape)

            if isinstance(layer, layers.ConvLSTM):
                conn_layer_idx = self.det_init_connections[layer_idx]
                cur_skip = ctx[conn_layer_idx][:, :self.n_ctx].contiguous()
                skip_layer = self.det_init_nets['layer_{}'.format(conn_layer_idx)]
                cur_skip = cur_skip.view(b, -1, cur_skip.shape[-2], cur_skip.shape[-1]).unsqueeze(1)
                cur_skip = skip_layer(cur_skip).squeeze(1)

                out = layer(out, torch.chunk(cur_skip, 2, 1))


            else:
                out = layer(out)

        out = torch.sigmoid(out)
        return out

    def forward(self, frames, config, use_prior, use_mean=False):

        bs, ts, cs, hs, ws = frames.shape

        stored_vars = []
        n_steps = config['n_steps']
        n_ctx = config['n_ctx']

        # Encode frames for latents and renderer
        pq_emb = self.latent_emb(frames)
        det_emb = self.det_emb(frames)

        # Get ctx emb for prior, posterior
        p_ctx = pq_emb[:, :n_ctx].contiguous()
        q_ctx = p_ctx

        # Get prior and posterior
        p_dists = self.prior(pq_emb[:, n_ctx - 1:-1].contiguous(), p_ctx, use_mean=use_mean)
        q_dists = self.posterior(pq_emb[:, n_ctx:].contiguous(), q_ctx, use_mean=use_mean)

        # Process prior and posterior
        ladj = None
        aux_p_dists = []
        for (means, logvars, z0, zk, ladj) in p_dists:
            means = means[:, -n_steps:].contiguous()
            logvars = logvars[:, -n_steps:].contiguous()
            z0 = z0[:, -n_steps:].contiguous()
            zk = zk[:, -n_steps:].contiguous()
            if ladj is not None:
                ladj = ladj[:, -n_steps:].contiguous()
            aux_p_dists.append([means, logvars, z0, zk, ladj])
        p_dists = aux_p_dists

        aux_q_dists = []
        for (means, logvars, z0, zk, ladj) in q_dists:
            means = means[:, -n_steps:].contiguous()
            logvars = logvars[:, -n_steps:].contiguous()
            z0 = z0[:, -n_steps:].contiguous()
            zk = zk[:, -n_steps:].contiguous()
            if ladj is not None:
                ladj = ladj[:, -n_steps:].contiguous()
            aux_q_dists.append([means, logvars, z0, zk, ladj])
        q_dists = aux_q_dists

        # Latent samples
        zs = []
        if use_prior:
            for (_, _, z0, _, _) in p_dists:
                zs.append(z0)
        else:
            for (_, _, _, zk, _) in q_dists:
                zs.append(zk)

        # Render frames
        preds = self.render(det_emb[-1][:, n_ctx - 1:-1], zs, det_emb)

        preds = preds[:, -n_steps:]

        return (preds, p_dists, q_dists), stored_vars


        
if __name__ == '__main__':
    pass
