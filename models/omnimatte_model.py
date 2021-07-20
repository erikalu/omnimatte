# Copyright 2021 Erika Lu
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
from third_party.models.base_model import BaseModel
from third_party.models.networks_lnr import MaskLoss, cal_alpha_reg
from . import networks
import numpy as np
import torch.nn.functional as F
import utils


class OmnimatteModel(BaseModel):
    """This class implements the layered neural rendering model for decomposing a video into layers."""
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='omnimatte')
        if is_train:
            parser.add_argument('--lambda_adj_reg', type=float, default=.001, help='regularizer for cam adjustment')
            parser.add_argument('--lambda_recon_flow', type=float, default=1., help='flow recon loss weight')
            parser.add_argument('--lambda_recon_warp', type=float, default=0., help='warped recon loss weight')
            parser.add_argument('--lambda_alpha_warp', type=float, default=0.005, help='alpha warping  loss weight')
            parser.add_argument('--lambda_alpha_l1', type=float, default=0.01, help='alpha L1 sparsity loss weight')
            parser.add_argument('--lambda_alpha_l0', type=float, default=0.005, help='alpha L0 sparsity loss weight')
            parser.add_argument('--alpha_l1_rolloff_epoch', type=int, default=200, help='turn off L1 alpha sparsity loss weight after this epoch')
            parser.add_argument('--lambda_mask', type=float, default=50, help='layer matting loss weight')
            parser.add_argument('--mask_thresh', type=float, default=0.02, help='turn off masking loss when error falls below this value')
            parser.add_argument('--mask_loss_rolloff_epoch', type=int, default=-1, help='decrease masking loss after this epoch; if <0, use mask_thresh instead')
            parser.add_argument('--cam_adj_epoch', type=int, default=0, help='when to start optimizing camera adjustment params')
            parser.add_argument('--jitter_rgb', type=float, default=0, help='amount of jitter to add to RGB')
            parser.add_argument('--jitter_epochs', type=int, default=0, help='number of epochs to jitter RGB')

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options
        """
        BaseModel.__init__(self, opt)
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['target_image', 'reconstruction', 'rgba_vis', 'alpha_vis', 'input_vis']
        self.model_names = ['Omnimatte']
        self.netOmnimatte = networks.define_omnimatte(opt.num_filters, opt.in_c, gpu_ids=self.gpu_ids)
        self.do_cam_adj = True
        if self.isTrain:
            self.setup_train(opt)

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def setup_train(self, opt):
        """Setup the model for training mode."""
        print('setting up model')
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['total', 'recon', 'alpha_reg', 'mask', 'recon_flow', 'recon_warp', 'alpha_warp', 'adj_reg']
        self.visual_names = ['target_image', 'reconstruction', 'rgba_vis', 'alpha_vis', 'input_vis', 'flow_vis', 'mask_vis']
        self.do_cam_adj = False
        self.criterionLoss = torch.nn.L1Loss()
        self.criterionLossMask = MaskLoss().to(self.device)
        self.lambda_mask = opt.lambda_mask
        self.lambda_adj_reg = opt.lambda_adj_reg
        self.lambda_recon_flow = opt.lambda_recon_flow
        self.lambda_recon_warp = opt.lambda_recon_warp
        self.lambda_alpha_warp = opt.lambda_alpha_warp
        self.lambda_alpha_l0 = opt.lambda_alpha_l0
        self.lambda_alpha_l1 = opt.lambda_alpha_l1
        self.mask_loss_rolloff_epoch = opt.mask_loss_rolloff_epoch
        self.jitter_rgb = opt.jitter_rgb
        self.optimizer = torch.optim.Adam(self.netOmnimatte.parameters(), lr=opt.lr)
        self.optimizers = [self.optimizer]

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.target_image = input['image'].to(self.device)
        if self.isTrain and self.jitter_rgb > 0:
            # add brightness jitter to rgb
            self.target_image += self.jitter_rgb * torch.randn(self.target_image.shape[0], 1, 1, 1).to(self.device)
            self.target_image = torch.clamp(self.target_image, -1, 1)
        self.input = input['input'].to(self.device)
        self.input_bg_flow = input['bg_flow'].to(self.device)
        self.input_bg_warp = input['bg_warp'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.flow_gt = input['flow'].to(self.device)
        self.flow_confidence = input['confidence'].to(self.device)
        self.jitter_grid = input['jitter_grid'].to(self.device)
        self.image_paths = input['image_path']
        self.index = input['index']

    def gen_crop_params(self, orig_h, orig_w, crop_size=256):
        """Generate random square cropping parameters."""
        starty = np.random.randint(orig_h - crop_size + 1)
        startx = np.random.randint(orig_w - crop_size + 1)
        endy = starty + crop_size
        endx = startx + crop_size
        return starty, endy, startx, endx

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        outputs = self.netOmnimatte(self.input, self.input_bg_flow, self.input_bg_warp, self.jitter_grid, self.index, self.do_cam_adj)
        # rearrange t, t+1 to batch dimension
        self.target_image = self.rearrange2batchdim(self.target_image)
        self.mask = self.rearrange2batchdim(self.mask)
        self.flow_confidence = self.rearrange2batchdim(self.flow_confidence)
        self.flow_gt = self.rearrange2batchdim(self.flow_gt)
        reconstruction_rgb = self.rearrange2batchdim(outputs['reconstruction_rgb'])
        self.reconstruction = reconstruction_rgb[:, :3]
        self.alpha_composite = reconstruction_rgb[:, 3:]
        self.reconstruction_warped = outputs['reconstruction_warped']
        self.alpha_warped = outputs['alpha_warped']
        self.bg_offset = self.rearrange2batchdim(outputs['bg_offset'])
        self.brightness_scale = self.rearrange2batchdim(outputs['brightness_scale'])
        self.reconstruction_flow = self.rearrange2batchdim(outputs['reconstruction_flow'])
        self.output_flow = self.rearrange2batchdim(outputs['layers_flow'])
        self.output_rgba = self.rearrange2batchdim(outputs['layers_rgba'])
        n_layers = self.output_rgba.shape[2]
        layers = self.output_rgba.clone()
        layers[:, -1, 0] = 1  # Background layer's alpha is always 1
        layers = torch.cat([layers[:, :, l] for l in range(n_layers)], -2)
        self.alpha_vis = layers[:, 3:]
        self.rgba_vis = layers#[:, :4]
        self.mask_vis = torch.cat([self.mask[:, l:l+1] for l in range(n_layers)], -2)
        self.input_vis = torch.cat([self.input[:, l, 3:6] for l in range(n_layers)], -2)  # TODO: visualize input flow
        self.input_vis -= self.input_vis.min()
        self.input_vis /= self.input_vis.max()
        self.flow_vis = torch.cat([self.output_flow[:, :, l] for l in range(n_layers)], -2)
        self.flow_vis = utils.tensor_flow_to_image(self.flow_vis[0].detach()).unsqueeze(0)  # batchsize 1

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.loss_recon = self.criterionLoss(self.reconstruction[:, :3], self.target_image)
        self.loss_total = self.loss_recon
        self.loss_alpha_reg = cal_alpha_reg(self.alpha_composite * .5 + .5, self.lambda_alpha_l1, self.lambda_alpha_l0)
        alpha_layers = self.output_rgba[:, 3]
        self.loss_mask = self.lambda_mask * self.criterionLossMask(alpha_layers, self.mask)
        self.loss_recon_flow = self.lambda_recon_flow * self.criterionLoss(self.flow_confidence * self.reconstruction_flow, self.flow_confidence * self.flow_gt)
        b_sz = self.target_image.shape[0]
        alpha_t = alpha_layers[:b_sz // 2]
        self.loss_alpha_warp = self.lambda_alpha_warp * self.criterionLoss(self.alpha_warped[:, 0], alpha_t)
        rgb_t = self.target_image[:b_sz // 2]
        self.loss_recon_warp = self.lambda_recon_warp * self.criterionLoss(self.reconstruction_warped, rgb_t)
        brightness_reg = self.criterionLoss(self.brightness_scale, torch.ones_like(self.brightness_scale))
        offset_reg = self.bg_offset.abs().mean()
        self.loss_adj_reg = self.lambda_adj_reg * (brightness_reg + offset_reg)
        self.loss_total += self.loss_alpha_reg + self.loss_mask + self.loss_recon_flow + self.loss_alpha_warp + self.loss_recon_warp + self.loss_adj_reg
        self.loss_total.backward()

    def rearrange2batchdim(self, tensor):
        n_c = tensor.shape[1]
        return torch.cat((tensor[:, :n_c // 2], tensor[:, n_c // 2:]))

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def update_lambdas(self, epoch):
        """Update loss weights based on current epochs and losses."""
        if epoch == self.opt.alpha_l1_rolloff_epoch:
            self.lambda_alpha_l1 = 0
        if self.mask_loss_rolloff_epoch >= 0:
            if epoch == 2*self.mask_loss_rolloff_epoch:
                self.lambda_mask = 0
        elif epoch > self.opt.epoch_count:
            if self.loss_mask < self.opt.mask_thresh * self.opt.lambda_mask:
                self.mask_loss_rolloff_epoch = epoch
                self.lambda_mask *= .1
        if epoch == self.opt.jitter_epochs:
            self.jitter_rgb = 0
        self.do_cam_adj = epoch >= self.opt.cam_adj_epoch

    def transfer_detail(self):
        """Transfer detail to layers."""
        residual = self.target_image - self.reconstruction
        transmission_comp = torch.zeros_like(self.target_image[:, 0:1])
        rgba_detail = self.output_rgba
        n_layers = self.output_rgba.shape[2]
        for i in range(n_layers - 1, 0, -1):  # Don't do detail transfer for background layer, due to ghosting effects.
            transmission_i = 1. - transmission_comp
            rgba_detail[:, :3, i] += transmission_i * residual
            alpha_i = self.output_rgba[:, 3:4, i] * .5 + .5
            transmission_comp = alpha_i + (1. - alpha_i) * transmission_comp
        self.rgba = torch.clamp(rgba_detail, -1, 1)

    def get_results(self):
        """Return results. This is different from get_current_visuals, which gets visuals for monitoring training.

        Returns a dictionary:
            original - - original frame
            recon - - reconstruction
            rgba_l* - - RGBA for each layer
            mask_l* - - mask for each layer
        """
        self.transfer_detail()
        results = {
            'reconstruction': self.reconstruction,
            'original': self.target_image,
            'reconstruction_flow': utils.tensor_flow_to_image(self.reconstruction_flow[0]).unsqueeze(0),  # batchsize 1
            'flow_gt': utils.tensor_flow_to_image(self.flow_gt[0]).unsqueeze(0),  # batchsize 1
            'bg_offset': utils.tensor_flow_to_image(self.bg_offset[0]).unsqueeze(0),  # batchsize 1
            'brightness_scale': self.brightness_scale - 1.
        }
        n_layers = self.rgba.shape[2]
        self.rgba[:, -1:, 0] = 1  # background layer's alpha is 1
        flow_layers = (self.rgba[:, -1:] * .5 + .5) * self.output_flow
        # Split layers
        for i in range(n_layers):
            results[f'mask_l{i}'] = self.mask[:, i:i+1]
            results[f'rgba_l{i}'] = self.rgba[:, :, i]
            results[f'flow_l{i}'] = utils.tensor_flow_to_image(flow_layers[0, :, i]).unsqueeze(0)  # batchsize 1
        return results
