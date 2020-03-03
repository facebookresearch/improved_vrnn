#
# Copyright (c) Facebook, Inc. and its affiliates.
#
"""
    s2s_big_hier.py
    ~~~~~~~~~~~~~~~~~~~~

    Seq2Seq model with Big ConvLSTM and hierarchical latents
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


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=False, norm_ch=4):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        self.actvn = nn.LeakyReLU(0.2)

        # Submodules
        self.conv_0 = nn.Sequential(
            nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1),
            nn.GroupNorm(norm_ch, self.fhidden)
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias),
            nn.GroupNorm(norm_ch, self.fout)
        )

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)


    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(self.actvn(x))
        dx = self.conv_1(self.actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


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

        self.sto_emb_net = nn.ModuleList([
            layers.DcConv(img_ch, n_hid, 4, 2, 1),
            layers.DcConv(n_hid, n_hid*2, 4, 2, 1),
            layers.DcConv(n_hid*2, n_hid*4, 4, 2, 1),
            layers.DcConv(n_hid*4, n_hid*8, 4, 2, 1),
            layers.DcConv(n_hid*8, enc_dim, 4, 1, 0, norm=partial(nn.GroupNorm, 1)),
        ])

        self.det_emb_net = nn.ModuleList([
            nn.Conv2d(img_ch, n_hid, 1),  
            ResnetBlock(n_hid, n_hid),
            nn.MaxPool2d(2, 2),
            ResnetBlock(n_hid*1, n_hid*2),
            ResnetBlock(n_hid*2, n_hid*2),
            nn.MaxPool2d(2, 2),
            ResnetBlock(n_hid*2, n_hid*4),
            ResnetBlock(n_hid*4, n_hid*4),
            nn.MaxPool2d(2, 2),
            ResnetBlock(n_hid*4, n_hid*4),
            ResnetBlock(n_hid*4, n_hid*8),
            nn.MaxPool2d(2, 2),
            ResnetBlock(n_hid*8, n_hid*8),
            ResnetBlock(n_hid*8, n_hid*8),
            nn.MaxPool2d(4, 1),
            ResnetBlock(n_hid*8, n_hid*8, norm_ch=1),
            ResnetBlock(n_hid*8, n_hid*8, norm_ch=1),
        ])

        mult = 1
        self.render_net = nn.ModuleList([
            layers.ConvLSTM(n_z + enc_dim, enc_dim),
            layers.DcUpConv(enc_dim, n_hid*8, 4, 1, 0),
            layers.ConvLSTM(n_hid*8 + n_hid*8, n_hid*8, norm=True),
            layers.DcUpConv(n_hid*8*mult, n_hid*8, 4, 2, 1),
            layers.ConvLSTM(n_hid*8 + n_hid*4, n_hid*8, norm=True),
            layers.DcUpConv(n_hid*8*mult, n_hid*4, 4, 2, 1),
            layers.ConvLSTM(n_hid*4, n_hid*4, norm=True),
            layers.DcUpConv(n_hid*4*mult, n_hid*2, 4, 2, 1),
            layers.ConvLSTM(n_hid*2, n_hid*2, norm=True),
            layers.DcUpConv(n_hid*2*mult, n_hid, 4, 2, 1),
            layers.ConvLSTM(n_hid, n_hid, norm=True),
            layers.DcConv(n_hid, n_hid, 3, 1, 1),
            layers.TemporalConv2d(n_hid, img_ch, 3, 1, 1),
        ])

        self.det_init_net = nn.Sequential(
            layers.DcConv(2*enc_dim*self.n_ctx, 2*enc_dim*self.n_ctx, 1),
            layers.TemporalConv2d(2*enc_dim*self.n_ctx, 2*enc_dim, 1),
            layers.TemporalNorm2d(1, 2*enc_dim),
        )

        self.prior_init_nets = nn.ModuleDict({
            'layer_4': nn.Sequential(
                layers.DcConv(enc_dim*self.n_ctx, enc_dim*self.n_ctx, 1),
                layers.TemporalConv2d(enc_dim*self.n_ctx, enc_dim*2, 1),
                layers.TemporalNorm2d(1, 2*enc_dim),
            ),
            'layer_3': nn.Sequential(
                layers.DcConv(n_hid*8*self.n_ctx, n_hid*8*self.n_ctx, 1),
                layers.TemporalConv2d(n_hid*8*self.n_ctx, n_hid*8*2, 1),
                layers.TemporalNorm2d(16, 2*n_hid*8),
            ),
            'layer_2': nn.Sequential(
                layers.DcConv(n_hid*4*self.n_ctx, n_hid*4*self.n_ctx, 1),
                layers.TemporalConv2d(n_hid*4*self.n_ctx, n_hid*4*2, 1),
                layers.TemporalNorm2d(16, 2*n_hid*4),
            ),
        })

        self.posterior_init_nets = nn.ModuleDict({
            'layer_4': nn.Sequential(
                layers.DcConv(enc_dim*self.n_ctx, enc_dim*self.n_ctx, 1),
                layers.TemporalConv2d(enc_dim*self.n_ctx, enc_dim*2, 1),
                layers.TemporalNorm2d(1, 2*enc_dim),
            ),
            'layer_3': nn.Sequential(
                layers.DcConv(n_hid*8*self.n_ctx, n_hid*8*self.n_ctx, 1),
                layers.TemporalConv2d(n_hid*8*self.n_ctx, n_hid*8*2, 1),
                layers.TemporalNorm2d(16, 2*n_hid*8),
            ),
            'layer_2': nn.Sequential(
                layers.DcConv(n_hid*4*self.n_ctx, n_hid*4*self.n_ctx, 1),
                layers.TemporalConv2d(n_hid*4*self.n_ctx, n_hid*4*2, 1),
                layers.TemporalNorm2d(16, 2*n_hid*4),
            ),
        })

        self.posterior_branches = nn.ModuleDict({
            'layer_4': nn.ModuleList([
                layers.TemporalConv2d(enc_dim, n_z, 1),
                layers.TemporalNorm2d(1, n_z),
                layers.ConvLSTM(n_z, enc_dim, norm=True),
                layers.TemporalConv2d(enc_dim, n_z*2, 1),
            ]),
            'layer_3': nn.ModuleList([
                layers.TemporalConv2d(n_hid*8, n_hid*8, 1),
                layers.TemporalNorm2d(16, n_hid*8),
                layers.ConvLSTM(n_hid*8, n_hid*8, norm=True),
                layers.TemporalConv2d(n_hid*8 + n_z, n_hid*8*2, 1),
            ]),
            'layer_2': nn.ModuleList([
                layers.TemporalConv2d(n_hid*4, n_hid*4, 1),
                layers.TemporalNorm2d(16, n_hid*4),
                layers.ConvLSTM(n_hid*4, n_hid*4, norm=True),
                layers.TemporalConv2d(n_hid*4 + n_hid*8 + n_z, n_hid*4*2, 1),
            ]),
        })

        self.prior_branches = nn.ModuleDict({
            'layer_4': nn.ModuleList([
                layers.TemporalConv2d(enc_dim, n_z, 1),
                layers.TemporalNorm2d(1, n_z),
                layers.ConvLSTM(n_z, enc_dim, norm=True),
                layers.TemporalConv2d(enc_dim, n_z*2, 1),
            ]),
            'layer_3': nn.ModuleList([
                layers.TemporalConv2d(n_hid*8, n_hid*8, 1),
                layers.TemporalNorm2d(16, n_hid*8),
                layers.ConvLSTM(n_hid*8, n_hid*8, norm=True),
                layers.TemporalConv2d(n_hid*8 + n_z, n_hid*8*2, 1),
            ]),
            'layer_2': nn.ModuleList([
                layers.TemporalConv2d(n_hid*4, n_hid*4, 1),
                layers.TemporalNorm2d(16, n_hid*4),
                layers.ConvLSTM(n_hid*4, n_hid*4, norm=True),
                layers.TemporalConv2d(n_hid*4 + n_hid*8 + n_z, n_hid*4*2, 1),
            ]),
        })

        # Connection list
        self.det_init_connections = {
            0: 16,
            2: 13,
            4: 10,
            6: 7,
            8: 4,
            10: 1,
        }

        # Connection branches
        self.det_init_nets = nn.ModuleDict({
            'layer_16': nn.Sequential(
                layers.DcConv(n_hid*8*self.n_ctx, n_hid*8*self.n_ctx, 1),
                layers.TemporalConv2d(n_hid*8*self.n_ctx, n_hid*8*2, 1),
                layers.TemporalNorm2d(1, n_hid*8*2)
            ),
            'layer_13': nn.Sequential(
                layers.DcConv(n_hid*8*self.n_ctx, n_hid*8*self.n_ctx, 3, 1, 1),
                layers.TemporalConv2d(self.n_ctx*n_hid*8, 2*n_hid*8, 1),
                layers.TemporalNorm2d(16, n_hid*8*2)
            ),
            'layer_10': nn.Sequential(
                layers.DcConv(n_hid*8*self.n_ctx, n_hid*8*self.n_ctx, 1),
                layers.TemporalConv2d(n_hid*8*self.n_ctx, n_hid*8*2, 1),
                layers.TemporalNorm2d(16, n_hid*8*2)
            ),
            'layer_7': nn.Sequential(
                layers.DcConv(n_hid*4*self.n_ctx, n_hid*4*self.n_ctx, 1),
                layers.TemporalConv2d(n_hid*4*self.n_ctx, n_hid*4*2, 1),
                layers.TemporalNorm2d(16, n_hid*8)
            ),
            'layer_4': nn.Sequential(
                layers.DcConv(n_hid*2*self.n_ctx, n_hid*2*self.n_ctx, 1),
                layers.TemporalConv2d(n_hid*2*self.n_ctx, n_hid*2*2, 1),
                layers.TemporalNorm2d(16, n_hid*4)
            ),
            'layer_1': nn.Sequential(
                layers.DcConv(n_hid*1*self.n_ctx, n_hid*1*self.n_ctx, 1),
                layers.TemporalConv2d(n_hid*1*self.n_ctx, n_hid*1*2, 1),
                layers.TemporalNorm2d(16, n_hid*2)
            ),
        })

        # Stochastic connection list
        # encoder -> renderer
        self.sto_branches = {
            4: 0,
            3: 2,
            2: 4,
        }
        # renderer -> encoder
        self.rend_sto_branches = {
            0: 0,
            2: 1, 
            4: 2,
        }

    def det_emb(self, x):
        out = x
        b, t, c, h, w = x.shape
        out = out.view(b*t, c, h, w)
        skips = []
        for layer_idx, layer in enumerate(self.det_emb_net):
            out = layer(out)

            cur_skip = out.view(b, t, -1, out.shape[-2], out.shape[-1])
            skips.append(cur_skip)

        return skips

    def sto_emb(self, x):
        out = x
        skips = []
        for layer_idx, layer in enumerate(self.sto_emb_net):
            out = layer(out)
            skips.append(out)

        return skips

    def prior(self, x, ctx, q_dists, use_prior, use_mean=False):
        dists = []
        sto_branches = sorted(self.sto_branches.keys(), reverse=True)

        for layer_idx in sto_branches:

            # print(layer_idx)

            # Find the corresopnding activations
            out = x[layer_idx][:, self.n_ctx - 1: -1].contiguous()
            cur_ctx = ctx[layer_idx][:, :self.n_ctx].contiguous()
            branch_layers = self.prior_branches['layer_{}'.format(layer_idx)]

            # Process the current branch
            for branch_layer_idx, layer in enumerate(branch_layers):

                # print(branch_layer_idx)

                if isinstance(layer, layers.ConvLSTM):
                    # Get initial condition
                    cur_ctx = cur_ctx.view(cur_ctx.shape[0], -1, cur_ctx.shape[-2], cur_ctx.shape[-1])
                    cur_ctx = cur_ctx.unsqueeze(1)
                    cur_ctx = self.prior_init_nets['layer_{}'.format(layer_idx)](cur_ctx)
                    cur_ctx = cur_ctx.squeeze(1)

                    # Forward LSTM
                    out = layer(out, torch.chunk(cur_ctx, 2, 1))

                # Handcrafted rules for integrating the different z's
                elif branch_layer_idx == 3:

                    if layer_idx == 4:
                        out = layer(out)

                    elif layer_idx == 3:
                        if use_prior:
                            z1 = dists[0][-2]
                        else:
                            z1 = q_dists[0][-2]
                        b, t, c, h, w = z1.shape
                        z1 = z1.view(b*t, c, h, w)
                        z1 = F.interpolate(z1, scale_factor=4)
                        z1 = z1.view(b, t, c, z1.shape[-2], z1.shape[-1])
                        out = torch.cat([out, z1], 2)
                        out = layer(out)

                    elif layer_idx == 2:
                        if use_prior:
                            z1 = dists[0][-2]
                        else:
                            z1 = q_dists[0][-2]
                        b, t, c, h, w = z1.shape
                        z1 = z1.view(b*t, c, h, w)
                        z1 = F.interpolate(z1, scale_factor=8)
                        z1 = z1.view(b, t, c, z1.shape[-2], z1.shape[-1])
                        if use_prior:
                            z2 = dists[1][-2]
                        else:
                            z2 = q_dists[1][-2]
                        b, t, c, h, w = z2.shape
                        z2 = z2.view(b*t, c, h, w)
                        z2 = F.interpolate(z2, scale_factor=2)
                        z2 = z2.view(b, t, c, z2.shape[-2], z2.shape[-1])
                        out = torch.cat([out, z1, z2], 2)
                        out = layer(out)

                else:
                    out = layer(out)

            # Compute distribution stats
            mean, var = torch.chunk(out, 2, 2)

            # Softplus var
            logvar = F.softplus(var).log()

            # Generate sample from this distribution
            z0 = flows.gaussian_rsample(mean, logvar, use_mean=use_mean)

            dists.append([mean, logvar, z0, z0, None])

        return dists


    def posterior(self, x, ctx, use_mean=False):
        dists = []
        sto_branches = sorted(self.sto_branches.keys(), reverse=True)

        for layer_idx in sto_branches:

            # print(layer_idx)

            # Find the corresopnding activations
            out = x[layer_idx][:, self.n_ctx:].contiguous()
            cur_ctx = ctx[layer_idx][:, :self.n_ctx].contiguous()
            branch_layers = self.posterior_branches['layer_{}'.format(layer_idx)]

            # Process the current branch
            for branch_layer_idx, layer in enumerate(branch_layers):

                # print(branch_layer_idx)

                if isinstance(layer, layers.ConvLSTM):
                    # Get initial condition
                    cur_ctx = cur_ctx.view(cur_ctx.shape[0], -1, cur_ctx.shape[-2], cur_ctx.shape[-1])
                    cur_ctx = cur_ctx.unsqueeze(1)
                    cur_ctx = self.posterior_init_nets['layer_{}'.format(layer_idx)](cur_ctx)
                    cur_ctx = cur_ctx.squeeze(1)

                    # Forward LSTM
                    out = layer(out, torch.chunk(cur_ctx, 2, 1))

                # Handcrafted rules for integrating the different z's
                elif branch_layer_idx == 3:

                    if layer_idx == 4:
                        out = layer(out)

                    elif layer_idx == 3:
                        z1 = dists[0][-2]
                        b, t, c, h, w = z1.shape
                        z1 = z1.view(b*t, c, h, w)
                        z1 = F.interpolate(z1, scale_factor=4)
                        z1 = z1.view(b, t, c, z1.shape[-2], z1.shape[-1])
                        out = torch.cat([out, z1], 2)
                        out = layer(out)

                    elif layer_idx == 2:
                        z1 = dists[0][-2]
                        b, t, c, h, w = z1.shape
                        z1 = z1.view(b*t, c, h, w)
                        z1 = F.interpolate(z1, scale_factor=8)
                        z1 = z1.view(b, t, c, z1.shape[-2], z1.shape[-1])
                        z2 = dists[1][-2]
                        b, t, c, h, w = z2.shape
                        z2 = z2.view(b*t, c, h, w)
                        z2 = F.interpolate(z2, scale_factor=2)
                        z2 = z2.view(b, t, c, z2.shape[-2], z2.shape[-1])
                        out = torch.cat([out, z1, z2], 2)
                        out = layer(out)

                else:
                    out = layer(out)

            # Compute distribution stats
            mean, var = torch.chunk(out, 2, 2)

            # Softplus var
            logvar = F.softplus(var).log()

            # Generate sample from this distribution
            z0 = flows.gaussian_rsample(mean, logvar, use_mean=use_mean)

            dists.append([mean, logvar, z0, z0, None])

        return dists

    def render(self, x, zs, ctx):
        b, t, c, h, w = x.shape

        out = x
        for layer_idx, layer in enumerate(self.render_net):

            # print(layer_idx, '->', out.shape)

            if layer_idx in self.rend_sto_branches:
                cur_zs = zs[self.rend_sto_branches[layer_idx]]
                out = torch.cat([out, cur_zs], 2)

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

        stored_vars = []
        n_steps = config['n_steps']
        n_ctx = config['n_ctx']

        # Encode frames for latents and renderer
        sto_emb = self.sto_emb(frames)
        det_emb = self.det_emb(frames)

        # Get prior and posterior
        q_dists = self.posterior(sto_emb, sto_emb, use_mean=use_mean)
        p_dists = self.prior(sto_emb, sto_emb, q_dists, use_prior, use_mean=use_mean)

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

        return (preds, p_dists, q_dists), stored_vars


        
if __name__ == '__main__':
    pass
