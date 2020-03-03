#
# Copyright (c) Facebook, Inc. and its affiliates.
#
from functools import partial
from itertools import chain

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

import layers
import flows
from vrnn_layers import *


class Model(nn.Module):

    def vrnn_arch(self, n_hid, n_z, enc_dim, img_ch):
        """Returns a dictionary with the structure of each component of VRNN."""

        arch = {}

        # Frame Embedding Net / Encoder net
        frame_emb = {}
        frame_emb['in_ch'] = [img_ch] + [n_hid*i for i in [1, 2, 4, 8, 8]]
        frame_emb['out_ch'] = [n_hid*i for i in [1, 2, 4, 8, 8, 8]]
        frame_emb['first_conv'] = [True, False, False, False, False, False]
        frame_emb['pool_ksize'] = [None, 2, 2, 2, 2, 4]
        frame_emb['pool_stride'] = [None, 2, 2, 2, 2, 1]
        arch['frame_emb'] = frame_emb

        # Renderer/Likelihood model
        renderer = {}
        renderer['in_ch'] = [n_hid*i for i in [8, 8, 8, 4, 2, 1]]
        renderer['hid_ch'] = [n_hid*i for i in [8, 8, 8, 4, 2, 1]]
        renderer['out_ch'] = [n_hid*i for i in [8, 8, 4, 2, 1, 1]]
        renderer['ksize'] = [4, 4, 4, 4, 4, 4]
        renderer['stride'] = [1, 2, 2, 2, 2, 2]
        renderer['padding'] = [0, 1, 1, 1, 1, 1] 
        renderer['upsample'] = [True, True, True, True, True, False]
        renderer['latent_idx'] = [0, None, None, None, None, None]
        arch['renderer'] = renderer

        # Prior/Posterior networks
        latent = {}
        latent['in_ch'] = [n_hid*8]
        latent['hid_ch'] = [n_hid*8]
        latent['out_ch'] = [n_z]
        latent['ctx_idx'] = [i for i, j in enumerate(reversed(renderer['latent_idx'])) if j is not None]
        latent['ctx_idx'] = list(reversed(latent['ctx_idx']))
        latent['resolution'] = [1]
        arch['latent'] = latent

        return arch


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

        # Get VRNN architecture
        self.arch = self.vrnn_arch(n_hid, n_z, enc_dim, img_ch)

        ### Define frame embedding network
        emb_net = []
        for i, _ in enumerate(self.arch['frame_emb']['in_ch']):
            arch = self.arch['frame_emb']
            inc = arch['in_ch'][i]
            outc = arch['out_ch'][i]
            pksize = arch['pool_ksize'][i]
            pstride = arch['pool_stride'][i]
            first_conv = arch['first_conv'][i]

            block = []

            if pksize is not None:
                block += [nn.MaxPool2d(pksize, pstride)]

            if first_conv:
                block += [nn.Conv2d(inc, outc, 1)]
            else:
                block += [ResnetBlock(inc, outc)]

            block += [ResnetBlock(outc, outc)]

            block = nn.Sequential(*block)
            emb_net += [block]

        self.emb_net = nn.ModuleList(emb_net)

        ### Define rendering network
        render_nets = []
        init_nets = []
        for i, _ in enumerate(self.arch['renderer']['in_ch']):
            arch = self.arch['renderer']
            inc = arch['in_ch'][i]
            hidc = arch['hid_ch'][i]
            outc = arch['out_ch'][i]
            ksize = arch['ksize'][i]
            padding = arch['padding'][i]
            stride = arch['stride'][i]
            upsample = arch['upsample'][i]
            init_inc = self.arch['frame_emb']['out_ch'][-(i + 1)]
            init_outc = hidc
            latent_idx = arch['latent_idx'][i]

            # Recompute ConvLSTM to have all the previous latents
            if latent_idx is not None:
                # import pdb; pdb.set_trace()
                # latent_ch = self.arch['latentl']['in_ch']
                latent_ch = self.arch['latent']['out_ch'][latent_idx]
                inc += latent_ch

            render_net = [ConvLSTM(inc, hidc, norm=True)]

            if upsample:
                render_net += [DcUpConv(hidc, outc, ksize, stride, padding)]
            else:
                render_net += [DcConv(hidc, outc, 3, 1, 1)]

            # Last layer of renderer
            if i == (len(arch['in_ch']) - 1):
                render_net += [TemporalConv2d(n_hid, img_ch, 3, 1, 1)]

            init_net = [
                DcConv(init_inc*self.n_ctx, init_inc*self.n_ctx, 1),
                TemporalConv2d(init_inc*self.n_ctx, init_outc*2, 1),
                TemporalNorm2d(1, init_outc*2),
            ]

            render_net = nn.ModuleList(render_net)
            render_nets.append(render_net)

            init_net = nn.Sequential(*init_net)
            init_nets.append(init_net)


        self.render_nets = nn.ModuleList(render_nets)
        self.init_nets = nn.ModuleList(init_nets)

        ### Define latent Net
        prior_init_nets = []
        posterior_init_nets = []
        prior_nets = []
        posterior_nets = []
        for i, _ in enumerate(self.arch['latent']['in_ch']):
            arch = self.arch['latent']
            inc = arch['in_ch'][i]
            hidc = arch['hid_ch'][i]
            outc = arch['out_ch'][i]

            # Compute previous channels
            prevc = sum(arch['out_ch'][:i])

            prior_net = []
            posterior_net = []
            prior_init_net = []
            posterior_init_net = []

            prior_net += [
                TemporalConv2d(inc, hidc, 1),
                TemporalNorm2d(1, hidc),
                ConvLSTM(hidc, hidc, norm=True),
                TemporalConv2d(hidc, outc*2, 1),
                TemporalNorm2d(1, outc*2),
            ]

            posterior_net += [
                TemporalConv2d(inc, hidc, 1),
                TemporalNorm2d(1, hidc),
                ConvLSTM(hidc, hidc, norm=True),
                TemporalConv2d(hidc + prevc, outc*2, 1),
                TemporalNorm2d(1, outc*2),
            ]

            prior_init_net += [
                DcConv(inc*self.n_ctx, inc*self.n_ctx, 1),
                TemporalConv2d(inc*self.n_ctx, hidc*2, 1),
                TemporalNorm2d(1, 2*hidc),

            ]

            posterior_init_net += [
                DcConv(inc*self.n_ctx, inc*self.n_ctx, 1),
                TemporalConv2d(inc*self.n_ctx, hidc*2, 1),
                TemporalNorm2d(1, 2*hidc),

            ]

            # Make modulelist
            prior_net = nn.ModuleList(prior_net)
            posterior_net = nn.ModuleList(posterior_net)
            prior_init_net = nn.Sequential(*prior_init_net)
            posterior_init_net = nn.Sequential(*posterior_init_net)

            # Append to the list of nets
            prior_nets.append(prior_net)
            posterior_nets.append(posterior_net)
            prior_init_nets.append(prior_init_net)
            posterior_init_nets.append(posterior_init_net)

        # Make module list
        self.prior_nets = nn.ModuleList(prior_nets)
        self.posterior_nets = nn.ModuleList(posterior_nets)
        self.prior_init_nets = nn.ModuleList(prior_init_nets)
        self.posterior_init_nets = nn.ModuleList(posterior_init_nets)

        # Init weights of last layers
        for block in chain(self.prior_nets, self.posterior_nets):
            nn.init.constant_(block[-1].model.weight, 0)
            nn.init.normal_(block[-1].model.bias, std=1e-3)

    def get_emb(self, x):
        out = x
        b, t, c, h, w = x.shape
        out = out.view(b*t, c, h, w)
        skips = []
        for layer_idx, layer in enumerate(self.emb_net):
            out = layer(out)

            cur_skip = out.view(b, t, -1, out.shape[-2], out.shape[-1])
            skips.append(cur_skip)

        return skips

    def prior(self, emb, q_dists, use_mean=False, scale_var=1.):
        dists = []

        for net_idx in range(len(self.prior_nets)):

            # print('[NET] PriorNet {}'.format(net_idx))

            # Find the corresopnding activations
            ctx_idx = self.arch['latent']['ctx_idx'][net_idx]
            out = emb[ctx_idx][:, self.n_ctx - 1: -1].contiguous()
            cur_ctx = emb[ctx_idx][:, :self.n_ctx].contiguous()
            branch_layers = self.prior_nets[net_idx]

            # Process the current branch
            for branch_layer_idx, layer in enumerate(branch_layers):

                # print('[NET] PriorNet {}/{}'.format(net_idx, branch_layer_idx))

                if isinstance(layer, ConvLSTM):
                    # Get initial condition
                    cur_ctx = cur_ctx.view(
                        cur_ctx.shape[0], 
                        -1, 
                        cur_ctx.shape[-2], 
                        cur_ctx.shape[-1]
                    )
                    cur_ctx = cur_ctx.unsqueeze(1)
                    cur_ctx = self.prior_init_nets[net_idx](cur_ctx)
                    cur_ctx = cur_ctx.squeeze(1)

                    # Forward LSTM
                    out = layer(out, torch.chunk(cur_ctx, 2, 1))

                else:
                    out = layer(out)

            # Compute distribution stats
            mean, var = torch.chunk(out, 2, 2)

            # Scale the variance
            var = var*scale_var

            # Softplus var
            logvar = F.softplus(var).log()

            # Generate sample from this distribution
            z0 = flows.gaussian_rsample(mean, logvar, use_mean=use_mean)

            dists.append([mean, logvar, z0, z0, None])

        return dists


    def posterior(self, emb, use_mean=False, scale_var=1.):
        dists = []

        for net_idx in range(len(self.posterior_nets)):

            # print('[NET] PosteriorNet {}'.format(net_idx))

            # Find the corresopnding activations
            ctx_idx = self.arch['latent']['ctx_idx'][net_idx]
            out = emb[ctx_idx][:, self.n_ctx:].contiguous()
            cur_ctx = emb[ctx_idx][:, :self.n_ctx].contiguous()
            branch_layers = self.posterior_nets[net_idx]

            # print('CTX IDX: ', ctx_idx, ' shape: ', cur_ctx.shape)
            # print(branch_layers)

            # Process the current branch
            for branch_layer_idx, layer in enumerate(branch_layers):

                # print('[NET] PosteriorNet {}/{}'.format(net_idx, branch_layer_idx))

                if isinstance(layer, ConvLSTM):

                    # Get initial condition
                    cur_ctx = cur_ctx.view(
                        cur_ctx.shape[0], 
                        -1, 
                        cur_ctx.shape[-2], 
                        cur_ctx.shape[-1]
                    )
                    cur_ctx = cur_ctx.unsqueeze(1)
                    cur_ctx = self.posterior_init_nets[net_idx](cur_ctx)
                    cur_ctx = cur_ctx.squeeze(1)

                    # Forward LSTM
                    out = layer(out, torch.chunk(cur_ctx, 2, 1))

                # Dense connectivity latent
                elif branch_layer_idx == 3:

                    # print('THIRD LAYER')

                    # Get current latent resolution
                    cur_res = self.arch['latent']['resolution'][net_idx]

                    # Accumulate previous z
                    prev_zs = []

                    for prev_z_idx in range(net_idx):

                        # Get previous z resolution
                        prev_res = self.arch['latent']['resolution'][prev_z_idx]

                        # Compute scaling factor
                        scaling_factor = cur_res//prev_res

                        # Interpolate previous z
                        z_prev = dists[prev_z_idx][-2]
                        b, t, c, h, w = z_prev.shape
                        z_prev = z_prev.view(b*t, c, h, w)
                        z_prev = F.interpolate(z_prev, scale_factor=scaling_factor)
                        z_prev = z_prev.view(b, t, c, z_prev.shape[-2], z_prev.shape[-1])
                        prev_zs.append(z_prev)

                    # Concatenate zs
                    prev_zs = torch.cat(prev_zs + [out], 2)
                    
                    # Forward through layer
                    out = layer(prev_zs)

                else:
                    out = layer(out)

            # Compute distribution stats
            mean, var = torch.chunk(out, 2, 2)

            # Scale the variance
            var = var*scale_var

            # Softplus var
            logvar = F.softplus(var).log()

            # Generate sample from this distribution
            z0 = flows.gaussian_rsample(mean, logvar, use_mean=use_mean)

            dists.append([mean, logvar, z0, z0, None])

        return dists

    def render(self, x, zs, ctx):
        b, t, c, h, w = x.shape

        out = x
        rev_ctx = list(reversed(ctx))
        for block_idx, block in enumerate(self.render_nets):

            # print('-'*80)
            # print('BLOCK {}\n{}'.format(block_idx, block))
            # print('INIT \n{}'.format(self.init_nets[block_idx]))
            # print('INPUT {}'.format(out.shape))
            # print('-'*80)

            # Incorporate latent
            latent_idx = self.arch['renderer']['latent_idx'][block_idx]
            if latent_idx is not None:
                cur_zs = zs[latent_idx]
                out = torch.cat([out, cur_zs], 2)

            # Get current context
            cur_ctx = rev_ctx[block_idx][:, :self.n_ctx].contiguous()
            cur_ctx = cur_ctx.view(b, -1, cur_ctx.shape[-2], cur_ctx.shape[-1])
            cur_ctx = cur_ctx.unsqueeze(1)

            # print('CTX SHAPE: {}'.format(cur_ctx.shape))
            cur_skip = self.init_nets[block_idx](cur_ctx).squeeze(1)

            for layer in block:

                if isinstance(layer, ConvLSTM):
                    out = layer(out, torch.chunk(cur_skip, 2, 1))
                else:
                    out = layer(out)

        out = torch.sigmoid(out)
        return out

    def forward(self, frames, config, use_prior, use_mean=False, scale_var=1.):

        stored_vars = []
        n_steps = config['n_steps']
        n_ctx = config['n_ctx']

        # Encode frames for latents and renderer
        emb = self.get_emb(frames)

        # Get prior and posterior
        q_dists = self.posterior(emb, use_mean=use_mean, scale_var=scale_var)
        p_dists = self.prior(emb, q_dists, use_mean=use_mean, scale_var=scale_var)

        # Latent samples
        zs = []
        if use_prior:
            for (_, _, z0, _, _) in p_dists:
                zs.append(z0)
        else:
            for (_, _, _, zk, _) in q_dists:
                zs.append(zk)

        # Render frames
        preds = self.render(emb[-1][:, n_ctx - 1:-1], zs, emb)

        return (preds, p_dists, q_dists), stored_vars


        
if __name__ == '__main__':
    pass
