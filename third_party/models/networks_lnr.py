# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import networks


###############################################################################
# Helper Functions
###############################################################################
def define_LNR(nf=64, texture_channels=16, texture_res=16, n_textures=25, gpu_ids=[]):
    """Create a layered neural renderer.

    Parameters:
        nf (int) -- the number of channels in the first/last conv layers
        texture_channels (int) -- the number of channels in the neural texture
        texture_res (int) -- the size of each individual texture map
        n_textures (int) -- the number of individual texture maps
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a layered neural rendering model.
    """
    net = LayeredNeuralRenderer(nf, texture_channels, texture_res, n_textures)
    return networks.init_net(net, gpu_ids)


def define_kp2uv(nf=64, gpu_ids=[]):
    """Create a keypoint-to-UV model.

    Parameters:
        nf (int) -- the number of channels in the first/last conv layers

    Returns a keypoint-to-UV model.
    """
    net = kp2uv(nf)
    return networks.init_net(net, gpu_ids)


def cal_alpha_reg(prediction, lambda_alpha_l1, lambda_alpha_l0):
    """Calculate the alpha regularization term.

    Parameters:
        prediction (tensor) - - composite of predicted alpha layers
        lambda_alpha_l1 (float) - - weight for the L1 regularization term
        lambda_alpha_l0 (float) - - weight for the L0 regularization term
    Returns the alpha regularization loss term
    """
#    assert prediction.max() <= 1.
#    assert prediction.min() >= 0.
    loss = 0.
    if lambda_alpha_l1 > 0:
        loss += lambda_alpha_l1 * torch.mean(prediction)
    if lambda_alpha_l0 > 0:
        # Pseudo L0 loss using a squished sigmoid curve.
        l0_prediction = (torch.sigmoid(prediction * 5.0) - 0.5) * 2.0
        loss += lambda_alpha_l0 * torch.mean(l0_prediction)
    return loss


##############################################################################
# Classes
##############################################################################
class MaskLoss(nn.Module):
    """Define the loss which encourages the predicted alpha matte to match the mask (trimap)."""

    def __init__(self):
        super(MaskLoss, self).__init__()
        self.loss = nn.L1Loss(reduction='none')

    def __call__(self, prediction, target):
        """Calculate loss given predicted alpha matte and trimap.

        Balance positive and negative regions. Exclude 'unknown' region from loss.

        Parameters:
            prediction (tensor) - - predicted alpha
            target (tensor) - - trimap

        Returns: the computed loss
        """
        mask_err = self.loss(prediction, target)
        pos_mask = F.relu(target)
        neg_mask = F.relu(-target)
        pos_mask_loss = (pos_mask * mask_err).sum() / (1 + pos_mask.sum())
        neg_mask_loss = (neg_mask * mask_err).sum() / (1 + neg_mask.sum())
        loss = .5 * (pos_mask_loss + neg_mask_loss)
        return loss


class ConvBlock(nn.Module):
    """Helper module consisting of a convolution, optional normalization and activation, with padding='same'."""

    def __init__(self, conv, in_channels, out_channels, ksize=4, stride=1, dil=1, norm=None, activation='relu'):
        """Create a conv block.

        Parameters:
            conv (convolutional layer) - - the type of conv layer, e.g. Conv2d, ConvTranspose2d
            in_channels (int) - - the number of input channels
            in_channels (int) - - the number of output channels
            ksize (int) - - the kernel size
            stride (int) - - stride
            dil (int) - - dilation
            norm (norm layer) - - the type of normalization layer, e.g. BatchNorm2d, InstanceNorm2d
            activation (str)  -- the type of activation: relu | leaky | tanh | none
        """
        super(ConvBlock, self).__init__()
        self.k = ksize
        self.s = stride
        self.d = dil
        self.conv = conv(in_channels, out_channels, ksize, stride=stride, dilation=dil)

        if norm is not None:
            self.norm = norm(out_channels)
        else:
            self.norm = None

        if activation == 'leaky':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = None

    def forward(self, x):
        """Forward pass. Compute necessary padding and cropping because pytorch doesn't have pad=same."""
        height, width = x.shape[-2:]
        if isinstance(self.conv, nn.modules.ConvTranspose2d):
            desired_height = height * self.s
            desired_width = width * self.s
            pady = 0
            padx = 0
        else:
            # o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
            # padding = .5 * (stride * (output-1) + (k-1)(d-1) + k - input)
            desired_height = height // self.s
            desired_width = width // self.s
            pady = .5 * (self.s * (desired_height - 1) + (self.k - 1) * (self.d - 1) + self.k - height)
            padx = .5 * (self.s * (desired_width - 1) + (self.k - 1) * (self.d - 1) + self.k - width)
        x = F.pad(x, [int(np.floor(padx)), int(np.ceil(padx)), int(np.floor(pady)), int(np.ceil(pady))])
        x = self.conv(x)
        if x.shape[-2] != desired_height or x.shape[-1] != desired_width:
            cropy = x.shape[-2] - desired_height
            cropx = x.shape[-1] - desired_width
            x = x[:, :, int(np.floor(cropy / 2.)):-int(np.ceil(cropy / 2.)),
                int(np.floor(cropx / 2.)):-int(np.ceil(cropx / 2.))]
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlock(nn.Module):
    """Define a residual block."""

    def __init__(self, channels, ksize=4, stride=1, dil=1, norm=None, activation='relu'):
        """Initialize the residual block, which consists of 2 conv blocks with a skip connection."""
        super(ResBlock, self).__init__()
        self.convblock1 = ConvBlock(nn.Conv2d, channels, channels, ksize=ksize, stride=stride, dil=dil, norm=norm,
                                    activation=activation)
        self.convblock2 = ConvBlock(nn.Conv2d, channels, channels, ksize=ksize, stride=stride, dil=dil, norm=norm,
                                    activation=None)

    def forward(self, x):
        identity = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        x += identity
        return x


class kp2uv(nn.Module):
    """UNet architecture for converting keypoint image to UV map.

    Same person UV map format as described in https://arxiv.org/pdf/1802.00434.pdf.
    """

    def __init__(self, nf=64):
        super(kp2uv, self).__init__(),
        self.encoder = nn.ModuleList([
            ConvBlock(nn.Conv2d, 3, nf, ksize=4, stride=2),
            ConvBlock(nn.Conv2d, nf, nf * 2, ksize=4, stride=2, norm=nn.InstanceNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, nf * 2, nf * 4, ksize=4, stride=2, norm=nn.InstanceNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, nf * 4, nf * 4, ksize=4, stride=2, norm=nn.InstanceNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, nf * 4, nf * 4, ksize=4, stride=2, norm=nn.InstanceNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, nf * 4, nf * 4, ksize=3, stride=1, norm=nn.InstanceNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, nf * 4, nf * 4, ksize=3, stride=1, norm=nn.InstanceNorm2d, activation='leaky')])

        self.decoder = nn.ModuleList([
            ConvBlock(nn.ConvTranspose2d, nf * 4 * 2, nf * 4, ksize=4, stride=2, norm=nn.InstanceNorm2d),
            ConvBlock(nn.ConvTranspose2d, nf * 4 * 2, nf * 4, ksize=4, stride=2, norm=nn.InstanceNorm2d),
            ConvBlock(nn.ConvTranspose2d, nf * 4 * 2, nf * 2, ksize=4, stride=2, norm=nn.InstanceNorm2d),
            ConvBlock(nn.ConvTranspose2d, nf * 2 * 2, nf, ksize=4, stride=2, norm=nn.InstanceNorm2d),
            ConvBlock(nn.ConvTranspose2d, nf * 2, nf, ksize=4, stride=2, norm=nn.InstanceNorm2d)])

        # head to predict body part class (25 classes - 24 body parts, 1 background.)
        self.id_pred = ConvBlock(nn.Conv2d, nf + 3, 25, ksize=3, stride=1, activation='none')
        # head to predict UV coordinates for every body part class
        self.uv_pred = ConvBlock(nn.Conv2d, nf + 3, 2 * 24, ksize=3, stride=1, activation='tanh')

    def forward(self, x):
        """Forward pass through UNet, handling skip connections.
        Parameters:
            x (tensor) - - rendered keypoint image, shape [B, 3, H, W]

        Returns:
            x_id (tensor): part id class probabilities
            x_uv (tensor): uv coordinates for each part id
        """
        skips = [x]
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i < 5:
                skips.append(x)
        for layer in self.decoder:
            x = torch.cat((x, skips.pop()), 1)
            x = layer(x)
        x = torch.cat((x, skips.pop()), 1)
        x_id = self.id_pred(x)
        x_uv = self.uv_pred(x)
        return x_id, x_uv


class LayeredNeuralRenderer(nn.Module):
    """Layered Neural Rendering model for video decomposition.

    Consists of neural texture, UNet, upsampling module.
    """

    def __init__(self, nf=64, texture_channels=16, texture_res=16, n_textures=25):
        super(LayeredNeuralRenderer, self).__init__(),
        """Initialize layered neural renderer.

        Parameters:
            nf (int) -- the number of channels in the first/last conv layers
            texture_channels (int) -- the number of channels in the neural texture
            texture_res (int) -- the size of each individual texture map
            n_textures (int) -- the number of individual texture maps
        """
        # Neural texture is implemented as 'n_textures' concatenated horizontally
        self.texture = nn.Parameter(torch.randn(1, texture_channels, texture_res, n_textures * texture_res))

        # Define UNet
        self.encoder = nn.ModuleList([
            ConvBlock(nn.Conv2d, texture_channels + 1, nf, ksize=4, stride=2),
            ConvBlock(nn.Conv2d, nf, nf * 2, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, nf * 2, nf * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, nf * 4, nf * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, nf * 4, nf * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, nf * 4, nf * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, nf * 4, nf * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky')])
        self.decoder = nn.ModuleList([
            ConvBlock(nn.ConvTranspose2d, nf * 4 * 2, nf * 4, ksize=4, stride=2, norm=nn.BatchNorm2d),
            ConvBlock(nn.ConvTranspose2d, nf * 4 * 2, nf * 4, ksize=4, stride=2, norm=nn.BatchNorm2d),
            ConvBlock(nn.ConvTranspose2d, nf * 4 * 2, nf * 2, ksize=4, stride=2, norm=nn.BatchNorm2d),
            ConvBlock(nn.ConvTranspose2d, nf * 2 * 2, nf, ksize=4, stride=2, norm=nn.BatchNorm2d),
            ConvBlock(nn.ConvTranspose2d, nf * 2, nf, ksize=4, stride=2, norm=nn.BatchNorm2d)])
        self.final_rgba = ConvBlock(nn.Conv2d, nf, 4, ksize=4, stride=1, activation='tanh')

        # Define upsampling block, which outputs a residual
        upsampling_ic = texture_channels + 5 + nf
        self.upsample_block = nn.Sequential(
            ConvBlock(nn.Conv2d, upsampling_ic, nf, ksize=3, stride=1, norm=nn.InstanceNorm2d),
            ResBlock(nf, ksize=3, stride=1, norm=nn.InstanceNorm2d),
            ResBlock(nf, ksize=3, stride=1, norm=nn.InstanceNorm2d),
            ResBlock(nf, ksize=3, stride=1, norm=nn.InstanceNorm2d),
            ConvBlock(nn.Conv2d, nf, 4, ksize=3, stride=1, activation='none'))

    def render(self, x):
        """Pass inputs for a single layer through UNet.

        Parameters:
            x (tensor) - - sampled texture concatenated with person IDs

        Returns RGBA for the input layer and the final feature maps.
        """
        skips = [x]
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i < 5:
                skips.append(x)
        for layer in self.decoder:
            x = torch.cat((x, skips.pop()), 1)
            x = layer(x)
        rgba = self.final_rgba(x)
        return rgba, x

    def forward(self, uv_map, id_layers, uv_map_upsampled=None, crop_params=None):
        """Forward pass through layered neural renderer.

        Steps:
        1. Sample from the neural texture using uv_map
        2. Input uv_map and id_layers into UNet
            2a. If doing upsampling, then pass upsampled inputs and results through upsampling module
        3. Composite RGBA outputs.

        Parameters:
            uv_map (tensor) - - UV maps for all layers, with shape [B, (2*L), H, W]
            id_layers (tensor) - - person ID for all layers, with shape [B, L, H, W]
            uv_map_upsampled (tensor) - - upsampled UV maps to input to upsampling module (if None, skip upsampling)
            crop_params
        """
        b_sz = uv_map.shape[0]
        n_layers = uv_map.shape[1] // 2
        texture = self.texture.repeat(b_sz, 1, 1, 1)
        composite = None
        layers = []
        sampled_textures = []
        for i in range(n_layers):
            # Get RGBA for this layer.
            uv_map_i = uv_map[:, i * 2:(i + 1) * 2, ...]
            uv_map_i = uv_map_i.permute(0, 2, 3, 1)
            sampled_texture = F.grid_sample(texture, uv_map_i, mode='bilinear', padding_mode='zeros')
            inputs = torch.cat([sampled_texture, id_layers[:, i:i + 1]], 1)
            rgba, last_feat = self.render(inputs)

            if uv_map_upsampled is not None:
                uv_map_up_i = uv_map_upsampled[:, i * 2:(i + 1) * 2, ...]
                uv_map_up_i = uv_map_up_i.permute(0, 2, 3, 1)
                sampled_texture_up = F.grid_sample(texture, uv_map_up_i, mode='bilinear', padding_mode='zeros')
                id_layers_up = F.interpolate(id_layers[:, i:i + 1], size=sampled_texture_up.shape[-2:],
                                             mode='bilinear')
                inputs_up = torch.cat([sampled_texture_up, id_layers_up], 1)
                upsampled_size = inputs_up.shape[-2:]
                rgba = F.interpolate(rgba, size=upsampled_size, mode='bilinear')
                last_feat = F.interpolate(last_feat, size=upsampled_size, mode='bilinear')
                if crop_params is not None:
                    starty, endy, startx, endx = crop_params
                    rgba = rgba[:, :, starty:endy, startx:endx]
                    last_feat = last_feat[:, :, starty:endy, startx:endx]
                    inputs_up = inputs_up[:, :, starty:endy, startx:endx]
                rgba_residual = self.upsample_block(torch.cat((rgba, inputs_up, last_feat), 1))
                rgba += .01 * rgba_residual
                rgba = torch.clamp(rgba, -1, 1)
                sampled_texture = sampled_texture_up

            # Update the composite with this layer's RGBA output
            if composite is None:
                composite = rgba
            else:
                alpha = rgba[:, 3:4] * .5 + .5
                composite = rgba * alpha + composite * (1.0 - alpha)
            layers.append(rgba)
            sampled_textures.append(sampled_texture)

        outputs = {
            'reconstruction': composite,
            'layers': torch.stack(layers, 1),
            'sampled texture': sampled_textures,  # for debugging
        }
        return outputs
