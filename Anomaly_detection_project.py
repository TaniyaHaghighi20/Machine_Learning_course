import os
import math
from argparse import ArgumentParser
from collections import namedtuple
from typing import Optional

import numpy as np
from einops import rearrange
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models
import pytorch_lightning as pl

# ============================================================
# 0. Basic building blocks (ResNet, Attention, Up/Down)
# ============================================================

def nonlinearity(x):
    # swish / SiLU
    return x * torch.sigmoid(x)

class Normalize(nn.Module):
    def __init__(self, num_channels, num_groups=32):
        super().__init__()
        num_groups = min(num_groups, num_channels)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.with_conv:
            return self.conv(x)
        else:
            return self.pool(x)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels=0, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        else:
            self.temb_proj = None

        self.norm2 = Normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.nin_shortcut = None

    def forward(self, x, temb=None):
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None and self.temb_proj is not None:
            h = h + self.temb_proj(temb)[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h)

        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k) * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw
        h_ = torch.bmm(v, w_).reshape(b, c, h, w)

        h_ = self.proj_out(h_)
        return x + h_

# ============================================================
# 1. VectorQuantizer2 (from your snippet)
# ============================================================

class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1)
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)

        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

# ============================================================
# 2. Encoder / Decoder (your versions, with our blocks)
# ============================================================

class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch * in_ch_mult[0]
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in,
                                  2 * z_channels if double_z else z_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)

    def forward(self, x):
        temb = None
        hs = [self.conv_in(x)]
        curr = hs[-1]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                curr = self.down[i_level].block[i_block](curr, temb)
                if len(self.down[i_level].attn) > 0:
                    curr = self.down[i_level].attn[i_block](curr)
                hs.append(curr)
            if i_level != self.num_resolutions - 1:
                curr = self.down[i_level].downsample(curr)
                hs.append(curr)

        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print(f"Working with z of shape {self.z_shape} = {np.prod(self.z_shape)} dimensions.")

        self.conv_in = nn.Conv2d(z_channels,
                                 block_in,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in,
                                  out_ch,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)

    def forward(self, z):
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

# ============================================================
# 3. ActNorm + small helpers (from your snippet)
# ============================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters())

class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError("Reverse init not allowed")
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h

# ============================================================
# 4. NLayerDiscriminator + LPIPS + VQLPIPSWithDiscriminator
# ============================================================

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        super().__init__()
        norm_layer = ActNorm if use_actnorm else nn.BatchNorm2d
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.main = nn.Sequential(*sequence)

    def forward(self, x):
        return self.main(x)

class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale

class NetLinLayer(nn.Module):
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super().__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)

class vgg16_lpips(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X); h_relu1_2 = h
        h = self.slice2(h); h_relu2_2 = h
        h = self.slice3(h); h_relu3_3 = h
        h = self.slice4(h); h_relu4_3 = h
        h = self.slice5(h); h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out

class LPIPS(nn.Module):
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.net = vgg16_lpips(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, input, target):
        in0, in1 = self.scaling_layer(input), self.scaling_layer(target)
        outs0, outs1 = self.net(in0), self.net(in1)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk] = normalize_tensor(outs0[kk])
            feats1[kk] = normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True)
               for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val = val + res[l]
        return val

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(F.softplus(-logits_real)) +
        torch.mean(F.softplus(logits_fake)))
    return d_loss

class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=1, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge", use_discriminator=True, perceptual_loss=None):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = perceptual_loss or LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.use_discriminator = use_discriminator

        if use_discriminator:
            self.discriminator = NLayerDiscriminator(
                input_nc=disc_in_channels,
                n_layers=disc_num_layers,
                use_actnorm=use_actnorm,
                ndf=disc_ndf
            ).apply(weights_init)
            self.discriminator_iter_start = disc_start
            self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
            print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
            self.disc_factor = disc_factor
            self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0], device=inputs.device)
        nll_loss = torch.mean(rec_loss)

        if optimizer_idx == 0:
            if not self.use_discriminator:
                d_weight = torch.tensor(0.0, device=inputs.device)
                disc_factor = torch.tensor(0.0, device=inputs.device)
                g_loss = torch.tensor(0.0, device=inputs.device)
                loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()
                log = {
                    f"{split}/total_loss": loss.detach().mean(),
                    f"{split}/quant_loss": codebook_loss.detach().mean(),
                    f"{split}/nll_loss": nll_loss.detach().mean(),
                    f"{split}/rec_loss": rec_loss.detach().mean(),
                    f"{split}/p_loss": p_loss.detach().mean(),
                    f"{split}/d_weight": d_weight.detach(),
                    f"{split}/disc_factor": disc_factor,
                    f"{split}/g_loss": g_loss.detach().mean(),
                }
                return loss, log

            if cond is None:
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                d_weight = torch.tensor(0.0, device=inputs.device)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()
            log = {
                f"{split}/total_loss": loss.detach().mean(),
                f"{split}/quant_loss": codebook_loss.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/p_loss": p_loss.detach().mean(),
                f"{split}/d_weight": d_weight.detach(),
                f"{split}/disc_factor": disc_factor,
                f"{split}/g_loss": g_loss.detach().mean(),
            }
            return loss, log

        if optimizer_idx == 1:
            if not self.use_discriminator:
                d_loss = torch.tensor(0.0, device=nll_loss.device)
                log = {
                    f"{split}/disc_loss": d_loss,
                    f"{split}/logits_real": torch.tensor(0.0, device=nll_loss.device),
                    f"{split}/logits_fake": torch.tensor(0.0, device=nll_loss.device),
                }
                return d_loss, log

            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            log = {
                f"{split}/disc_loss": d_loss.detach().mean(),
                f"{split}/logits_real": logits_real.detach().mean(),
                f"{split}/logits_fake": logits_fake.detach().mean(),
            }
            return d_loss, log

# ============================================================
# 5. VQModel (LightningModule)
# ============================================================

class VQModel(pl.LightningModule):
    def __init__(
        self,
        ddconfig,
        n_embed,
        embed_dim,
        learning_rate=4.5e-6,
        disc_start=10000,
        image_key="image",
        sane_index_shape=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.image_key = image_key
        self.learning_rate = learning_rate

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer2(
            n_e=n_embed,
            e_dim=embed_dim,
            beta=0.25,
            remap=None,
            sane_index_shape=sane_index_shape
        )
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.loss = VQLPIPSWithDiscriminator(
            disc_start=disc_start,
            disc_in_channels=ddconfig["out_ch"],
            perceptual_weight=1.0,
            disc_factor=1.0,
            disc_weight=1.0,
            pixelloss_weight=1.0,
            codebook_weight=1.0,
            use_discriminator=True,
        )

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, x):
        quant, diff, _ = self.encode(x)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if x.ndim == 3:  # HWC
            x = x[None, ...]
        if x.shape[-1] in [1, 3]:  # HWC -> BCHW
            x = x.permute(0, 3, 1, 2).contiguous()
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            aeloss, log_dict_ae = self.loss(
                qloss, x, xrec, optimizer_idx, self.global_step,
                last_layer=self.get_last_layer(), split="train"
            )
            self.log("train/aeloss", aeloss, prog_bar=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            discloss, log_dict_disc = self.loss(
                qloss, x, xrec, optimizer_idx, self.global_step,
                last_layer=self.get_last_layer(), split="train"
            )
            self.log("train/discloss", discloss, prog_bar=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        aeloss, log_dict_ae = self.loss(
            qloss, x, xrec, 0, self.global_step,
            last_layer=self.get_last_layer(), split="val"
        )
        discloss, log_dict_disc = self.loss(
            qloss, x, xrec, 1, self.global_step,
            last_layer=self.get_last_layer(), split="val"
        )

        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=False)
        self.log("val/aeloss", aeloss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=False)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quantize.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()),
            lr=lr, betas=(0.5, 0.9)
        )
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(),
            lr=lr, betas=(0.5, 0.9)
        )
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

# ============================================================
# 6. OCT Dataset + DataModule
# ============================================================

class OCTFolderDataset(Dataset):
    def __init__(self, root_dir: str, image_size: int = 256):
        self.root_dir = root_dir
        self.paths = []
        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        for r, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(exts):
                    self.paths.append(os.path.join(r, f))
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),                      # [0,1]
            T.Normalize(mean=[0.5], std=[0.5]) # [-1,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("L")
        img = self.transform(img)  # C,H,W
        return {"image": img.permute(1, 2, 0)}  # H,W,C

class OCTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        image_size: int = 256,
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = OCTFolderDataset(self.train_dir, image_size=self.image_size)
        self.val_dataset = OCTFolderDataset(self.val_dir, image_size=self.image_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

# ============================================================
# 7. ddconfig helper + main()
# ============================================================

def make_ddconfig(image_size: int = 256):
    return {
        "double_z": False,
        "z_channels": 256,
        "resolution": image_size,
        "in_channels": 1,
        "out_ch": 1,
        "ch": 128,
        "ch_mult": [1, 1, 2, 2, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [16],
        "dropout": 0.0,
    }

def main():
    parser = ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="data/oct/train")
    parser.add_argument("--val_dir", type=str, default="data/oct/val")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--disc_start", type=int, default=10000)
    args = parser.parse_args()

    pl.seed_everything(42)

    dm = OCTDataModule(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    ddconfig = make_ddconfig(image_size=args.image_size)

    model = VQModel(
        ddconfig=ddconfig,
        n_embed=256,
        embed_dim=256,
        learning_rate=args.learning_rate,
        disc_start=args.disc_start,
        image_key="image",
        sane_index_shape=False,
    )

    trainer = pl.Trainer(
        accelerator="gpu" if args.gpus > 0 and torch.cuda.is_available() else "cpu",
        devices=args.gpus if torch.cuda.is_available() else 1,
        max_epochs=args.max_epochs,
        log_every_n_steps=50,
        default_root_dir="vqgan_oct_logs",
        check_val_every_n_epoch=1,
    )

    trainer.fit(model, dm)

if __name__ == "__main__":
    main()
