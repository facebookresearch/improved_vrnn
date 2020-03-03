#
# Copyright (c) Facebook, Inc. and its affiliates.
#
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn


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


class DcConv(nn.Module):
    def __init__(self, nin, nout, kernel, stride=1, padding=0,
        conv=nn.Conv2d, norm=partial(nn.GroupNorm, 4), act=nn.LeakyReLU(0.2, inplace=True)
    ):
        super().__init__()

        model = [conv(nin, nout, kernel, stride=stride, padding=padding)]
        if norm is not None:
            model.append(norm(nout))
        if act is not None:
            model.append(act)

        self.main = nn.Sequential(*model)

    def forward(self, input):
        b, t, c, h, w = input.shape
        out = input.view(b*t, c, h, w)
        out = self.main(out)
        out = out.view(b, -1, out.shape[-3], out.shape[-2], out.shape[-1])
        return out


class DcUpConv(nn.Module):
    def __init__(self, nin, nout, kernel, stride=1, padding=0,
        conv=nn.ConvTranspose2d, norm=partial(nn.GroupNorm, 4), act=nn.LeakyReLU(0.2, inplace=True)
    ):
        super().__init__()

        if act is not None:
            self.main = nn.Sequential(
                conv(nin, nout, kernel, stride=stride, padding=padding),
                norm(nout),
                act,
            )
        else:
            self.main = nn.Sequential(
                conv(nin, nout, kernel, stride=stride, padding=padding),
                norm(nout),
            )


    def forward(self, input):
        b, t, c, h, w = input.shape
        out = input.view(b*t, c, h, w)
        out = self.main(out)
        out = out.view(b, -1, out.shape[-3], out.shape[-2], out.shape[-1])
        return out


class ConvLSTM(nn.Module):
    def __init__(self, in_ch, hid_ch, norm=False, reverse=False):
        super().__init__()

        if not norm:
            self.model = ConvLSTMCell(in_ch, hid_ch)
            # raise NotImplementedError
        else:
            self.model = NormConvLSTMCell(in_ch, hid_ch)

        self.in_ch = in_ch
        self.hid_ch = hid_ch
        self.reverse = reverse

    def forward(self, x, cond=None):

        b, t, c, h, w = x.shape
        if cond is None:
            cond = torch.zeros(b, self.hid_ch*2, h, w, device=x.device, dtype=x.dtype)
            prev_state = torch.chunk(cond, 2, 1)
        else:
            prev_state = cond

        outs = []
        loop_range = range(t)
        if self.reverse: 
            loop_range = reversed(range(t))

        for timestep in loop_range:
            cur_in = x[:, timestep]
            out, prev_state = self.model(cur_in, prev_state)
            outs.append(out)

        if self.reverse:
            outs = list(reversed(outs))

        return torch.stack(outs, 1)


class NormConvLSTMCell(nn.Module):
    """
    Convolutional LSTM
    """
    def __init__(self, in_ch, hid_ch, kernel_size=3, padding=1):
        super().__init__()

        self.in_ch = in_ch 
        self.hid_ch = hid_ch
        self.kernel_size = kernel_size
        self.padding = padding

        self.ih_gates = nn.Sequential(
            nn.Conv2d(in_ch, 4*hid_ch, kernel_size, padding=padding),
            nn.GroupNorm(16, 4*hid_ch),
        )

        self.hh_gates = nn.Sequential(
            nn.Conv2d(hid_ch, 4*hid_ch, kernel_size, padding=padding),
            nn.GroupNorm(16, 4*hid_ch),
        )

        self.c_norm = nn.GroupNorm(16, hid_ch)

    def forward(self, input_, prev_state):

        # Get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            # prev_state = (
            #     torch.zeros(state_size),
            #     torch.zeros(state_size)
            # )
            prev_state = torch.zeros(2*state_size, dtype=input_.dtype, device=input_.device)
            prev_state = torch.chunk(prev_state, 2, 0)

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        ih_gates = self.ih_gates(input_)
        hh_gates = self.hh_gates(prev_hidden)
        out_gates = ih_gates + hh_gates

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = out_gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        cell = self.c_norm(cell)
        hidden = out_gate * torch.tanh(cell)

        return hidden, (hidden, cell)


class TemporalConv2d(nn.Module):
    """
    Applies a 2D convolution over a 5D tensor (using reshapes)

    Args:
        x: B x T x C x H x W input.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        aux = {}
        if 'residual' in kwargs:
            for k, v in kwargs.items():
                if k != 'residual':
                    aux[k] = v
                else:
                    self.residual = kwargs['residual']
        else:
            aux = kwargs
            self.residual = False

        self.model = nn.Conv2d(*args, **aux)

    def forward(self, x):
        orig_x = x
        if len(x.shape) == 5:
            b, t, c, h, w = x.shape
            x = x.contiguous()
            x = x.view(b*t, c, h, w)
            out = self.model(x)
            out = out.view(b, t, out.size(-3), out.size(-2), out.size(-1))
        else:
            out = self.model(x)

        if self.residual:
            out = out + orig_x

        return out


class TemporalNorm2d(nn.Module):
    def __init__(self, n_groups, n_ch):
        super().__init__()
        self.model = nn.GroupNorm(n_groups, n_ch)

    def forward(self, x):
        b, t, c, h, w = x.shape
        out = x.view(b*t, c, h, w)
        out = self.model(out)
        out = out.view(b, t, out.shape[-3], out.shape[-2], out.shape[-1])

        return out


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM
    """
    def __init__(self, in_ch, hid_ch, kernel_size=3, padding=1):
        super().__init__()

        self.in_ch = in_ch 
        self.hid_ch = hid_ch
        self.kernel_size = kernel_size
        self.padding = padding

        self.gates = nn.Conv2d(
            in_ch + hid_ch, 
            4*hid_ch, 
            kernel_size, 
            padding=padding,
        )

        # self.gates = nn.Sequential(
        #     nn.Conv2d(in_ch + hid_ch, 4*hid_ch, kernel_size, padding=padding),
        #     nn.GroupNorm(16, 4*hid_ch)
        # )
        
        # self.gates = nn.Sequential(
        #     nn.Conv2d(in_ch + hid_ch, 4*hid_ch, kernel_size, padding=padding),
        #     nn.GroupNorm(4, 4*hid_ch),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(4*hid_ch, 4*hid_ch, kernel_size, padding=padding),
        #     nn.GroupNorm(4, 4*hid_ch),
        # )

    def forward(self, input_, prev_state):

        # Get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            # prev_state = (
            #     torch.zeros(state_size),
            #     torch.zeros(state_size)
            # )
            prev_state = torch.zeros(2*state_size, dtype=input_.dtype, device=input_.device)
            prev_state = torch.chunk(prev_state, 2, 0)

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        out_gates = self.gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = out_gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, (hidden, cell)
