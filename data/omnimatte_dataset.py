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


import cv2
from third_party.data.base_dataset import BaseDataset
from third_party.data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import glob
import torch
import numpy as np
import json
import utils


def load_and_process_image(im_path, mode='RGB', size=None):
    """Read image file and return as tensor in range [-1, 1]."""
    image = Image.open(im_path).convert(mode)
    if size is not None:
        image = image.resize(size)
    image = transforms.ToTensor()(image)
    image = 2 * image - 1
    return image


def load_and_resize_flow(flow_path, width=None, height=None):
    flow = torch.from_numpy(utils.readFlow(flow_path)).permute(2, 0, 1)
    flow = utils.resize_flow(flow, width, height)
    return flow


def apply_transform(data, params, interp_mode='bilinear'):
    """Apply the transform to the data tensor."""
    tensor_size = params['jitter size'].tolist()
    crop_pos = params['crop pos']
    crop_size = params['crop size']
    orig_shape = data.shape
    if len(orig_shape) < 4:
        data = F.interpolate(data.unsqueeze(0), size=tensor_size, mode=interp_mode).squeeze(0)
    else:
        data = F.interpolate(data, size=tensor_size, mode=interp_mode)
    data = data[..., crop_pos[0]:crop_pos[0] + crop_size[0], crop_pos[1]:crop_pos[1] + crop_size[1]]
    return data


def transform2h(x, y, m):
    """Applies 2d homogeneous transformation."""
    A = torch.matmul(m, torch.stack([x, y, torch.ones(len(x))]))
    xt = A[0, :] / A[2, :]
    yt = A[1, :] / A[2, :]
    return xt, yt


class OmnimatteDataset(BaseDataset):
    """A dataset class for video layers.

    It assumes that the directory specified by 'dataroot' contains metadata.json, and the directories iuv, rgb_256, and rgb_512.
    The 'iuv' directory should contain directories named 01, 02, etc. for each layer, each containing per-frame UV images.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--height', type=int, default=256, help='image height')
        parser.add_argument('--width', type=int, default=448, help='image width')
        parser.add_argument('--in_c', type=int, default=16, help='# input channels')
        parser.add_argument('--jitter_rate', type=float, default=0.75, help='')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        rgbdir = os.path.join(opt.dataroot, 'rgb')
        maskdir = os.path.join(opt.dataroot, 'mask')
        self.image_paths = sorted(make_dataset(rgbdir, opt.max_dataset_size))
        self.image_paths = self.image_paths[:-1]
        n_images = len(self.image_paths)
        layers = sorted(os.listdir(maskdir))
        layers = [l for l in layers if l.isdigit()]
        self.mask_paths = []
        for l in layers:
            layer_mask_paths = sorted(make_dataset(os.path.join(maskdir, l), n_images))
            if len(layer_mask_paths) != n_images:
                print(f'UNEQUAL NUMBER OF IMAGES AND MASKS: {len(layer_mask_paths)} and {n_images}')
            self.mask_paths.append(layer_mask_paths)
        self.flow_paths = sorted(glob.glob(os.path.join(opt.dataroot, 'flow', '*.flo')))
        self.confidence_paths = sorted(make_dataset(os.path.join(opt.dataroot, 'confidence')))

        zbar_path = os.path.join(opt.dataroot, 'zbar.pth')
        if not os.path.exists(zbar_path):
            zbar = torch.randn(1, opt.in_c - 3, opt.height // 16, opt.width // 16)
            torch.save(zbar, zbar_path)
        else:
            zbar = torch.load(zbar_path)
        self.Zbar = zbar
        self.Zbar_up = F.interpolate(zbar, (opt.height, opt.width), mode='bilinear')

        self.composite_order = [tuple(range(1, 1 + len(layers)))] * n_images
        self.init_homographies(os.path.join(opt.dataroot, 'homographies.txt'), n_images)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains the outputs of get_transformed_item for two consecutive frames, concatenated
            channelwise.
        """
        transform_params = self.get_params(do_jitter=self.opt.phase == 'train', jitter_rate=self.opt.jitter_rate)
        data_t1 = self.get_transformed_item(index, transform_params)
        data_t2 = self.get_transformed_item(index + 1, transform_params)
        data = {k : torch.cat((data_t1[k], data_t2[k]), -3) for k in data_t1 if k not in ['image_path', 'index']}
        data['image_path'] = data_t1['image_path']
        data['index'] = torch.cat((data_t1['index'], data_t2['index']))
        return data

    def get_transformed_item(self, index, transform_params):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains:
            input - - input to the Omnimatte model
            mask - - object trimaps
            flow - - flow inside object region
            bg_flow - - flow for background layer
            bg_warp - - warping grid used to sample from unwrapped background
            confidence - - flow confidence map
            jitter_grid - - sampling grid that was used for cropping / resizing
            image_path - - path to frame
            index - - frame index
        """
        # Read the target image.
        image_path = self.image_paths[index]
        target_image = load_and_process_image(image_path)

        nl = len(self.composite_order[index]) + 1

        # Create layer inputs by concatenating mask, flow, and background UVs.
        # Read the layer masks.
        masks = [load_and_process_image(self.mask_paths[l - 1][index], mode='L', size=(self.opt.width, self.opt.height)) for l in
                        self.composite_order[index]]
        mask_h, mask_w = masks[0].shape[-2:]
        masks = torch.stack(masks)  # L-1, 1, H, W
        binary_masks = (masks > 0).float()

        # Read flow
        if index >= len(self.flow_paths):
            # for last frame just use zero flow
            flow = torch.zeros(2, self.opt.height, self.opt.width)
        else:
            flow = load_and_resize_flow(self.flow_paths[index], mask_w, mask_h)

        # Create bg warp field from homographies.
        bg_warp = self.get_background_uv(index, mask_w, mask_h) * 2 - 1

        # Create the background Z_t from homographies.
        background_Zt = F.grid_sample(self.Zbar, bg_warp.permute(1, 2, 0).unsqueeze(0))  # C, H, W
        background_Zt = background_Zt.repeat(nl - 1, 1, 1, 1)

        # Build inputs from masks, flow, background UVs, and unwrapped bg
        input_flow = flow.unsqueeze(0).repeat(nl - 1, 1, 1, 1)
        input_flow *= binary_masks
        pids = torch.Tensor(self.composite_order[index]).view(-1, 1, 1, 1) * binary_masks  # L-1, 1, H, W
        inputs = torch.cat((pids, input_flow, background_Zt), 1)  # L-1, 16, H, W
        background_input = torch.cat((torch.zeros(1, 3, mask_h, mask_w), self.Zbar_up), 1)
        inputs = torch.cat((background_input, inputs))  # L, 16, H, W

        # Create bg flow and read confidence
        if index == len(self):
            # for last frame just set to zero (not used) 
            bg_flow = torch.zeros(2, mask_h, mask_w)
            confidence = torch.zeros(1, mask_h, mask_w)
        else:
            bg_flow = self.get_background_flow(index, mask_w, mask_h)  # 2, H, W
            confidence = load_and_process_image(self.confidence_paths[index], mode='L', size=(self.opt.width, self.opt.height)) * .5 + .5  # [0, 1] range
            confidence *= (binary_masks.sum(0) > 0).float()

        masks = masks[:, 0]
        masks = torch.stack([self.mask2trimap(masks[i]) for i in range(nl - 1)])
        masks = torch.cat((torch.zeros_like(masks[0:1]), masks))  # add bg mask

        jitter_grid = self.create_grid(mask_w, mask_h)
        jitter_grid = apply_transform(jitter_grid, transform_params, 'bilinear')
        masks = apply_transform(masks, transform_params, 'bilinear')
        inputs = apply_transform(inputs, transform_params, 'bilinear')
        bg_warp = apply_transform(bg_warp, transform_params, 'bilinear')
        confidence = apply_transform(confidence, transform_params, 'bilinear')
        # when applying transform to flow, also need to rescale
        scale_w = transform_params['jitter size'][1] / mask_w
        scale_h = transform_params['jitter size'][0] / mask_h
        inputs[:, 1] *= scale_w
        inputs[:, 2] *= scale_h
        flow = apply_transform(flow, transform_params, 'bilinear')
        flow[0] *= scale_w
        flow[1] *= scale_h
        bg_flow = apply_transform(bg_flow, transform_params, 'bilinear')
        bg_flow[0] *= scale_w
        bg_flow[1] *= scale_h

        image_transform_params = transform_params
        target_image = apply_transform(target_image, image_transform_params, 'bilinear')

        data = {'image': target_image,
                'input': inputs,
                'mask': masks,
                'flow': flow,
                'bg_flow': bg_flow,
                'bg_warp': bg_warp,
                'confidence': confidence,
                'jitter_grid': jitter_grid,
                'image_path': image_path,
                'index': torch.Tensor([index]).long()}
        return data

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths) - 1

    def get_params(self, do_jitter=False, jitter_rate=0.75):
        """Get transformation parameters."""
        if do_jitter:
            if np.random.uniform() > jitter_rate:
                scale = 1.
            else:
                scale = np.random.uniform(1, 1.25)
            jitter_size = (scale * np.array([self.opt.height, self.opt.width])).astype(np.int)
            start1 = np.random.randint(jitter_size[0] - self.opt.height + 1)
            start2 = np.random.randint(jitter_size[1] - self.opt.width + 1)
        else:
            jitter_size = np.array([self.opt.height, self.opt.width])
            start1 = 0
            start2 = 0
        crop_pos = np.array([start1, start2])
        crop_size = np.array([self.opt.height, self.opt.width])
        return {'jitter size': jitter_size, 'crop pos': crop_pos, 'crop size': crop_size}

    def init_homographies(self, homography_path, n_images):
        """Read homography file and set up homography data."""
        with open(homography_path) as f:
            h_data = f.readlines()
        h_scale = h_data[0].rstrip().split(' ')
        self.h_scale_x = int(h_scale[1])
        self.h_scale_y = int(h_scale[2])
        h_bounds = h_data[1].rstrip().split(' ')
        self.h_bounds_x = [float(h_bounds[1]), float(h_bounds[2])]
        self.h_bounds_y = [float(h_bounds[3]), float(h_bounds[4])]
        homographies = h_data[2:2 + n_images]
        homographies = [torch.from_numpy(np.array(line.rstrip().split(' ')).astype(np.float32).reshape(3, 3)) for line
                        in
                        homographies]
        self.homographies = homographies

    def create_grid(self, w, h):
        ramp_u = torch.linspace(-1, 1, steps=w).unsqueeze(0).repeat(h, 1)
        ramp_v = torch.linspace(-1, 1, steps=h).unsqueeze(-1).repeat(1, w)
        grid = torch.stack([ramp_u, ramp_v], 0)
        return grid

    def get_background_flow(self, index, w, h):
        """Return background layer UVs at 'index' (output range [0, 1])."""
        if hasattr(self, 'homographies'):
            ramp_u = torch.linspace(0, self.h_scale_x, steps=w).unsqueeze(0).repeat(h, 1)
            ramp_v = torch.linspace(0, self.h_scale_y, steps=h).unsqueeze(-1).repeat(1, w)
            ramp_ = torch.stack([ramp_u, ramp_v], 0)
            ramp = ramp_.reshape(2, -1)
            H_0 = self.homographies[index]
            H_1 = self.homographies[index + 1]
            # apply homography
            [xt, yt] = transform2h(ramp[0], ramp[1], torch.inverse(H_0))
            [xt, yt] = transform2h(xt, yt, H_1)
            # restore shape
            flow = torch.stack([xt.reshape(h, w), yt.reshape(h, w)], 0)
            flow -= ramp_
            # # scale from world to [-1, 1]
            # flow[0] /= .5 * self.h_scale_x
            # flow[1] /= .5 * self.h_scale_y
            # scale from world to image space
            flow[0] *= self.opt.width / self.h_scale_x
            flow[1] *= self.opt.height / self.h_scale_y
        else:
            flow = torch.zeros(2, h, w)
        return flow

    def get_background_uv(self, index, w, h):
        """Return background layer UVs at 'index' (output range [0, 1])."""
        ramp_u = torch.linspace(0, 1, steps=w).unsqueeze(0).repeat(h, 1)
        ramp_v = torch.linspace(0, 1, steps=h).unsqueeze(-1).repeat(1, w)
        ramp = torch.stack([ramp_u, ramp_v], 0)
        if hasattr(self, 'homographies'):
            # scale to [0, orig width/height]
            ramp[0] *= self.h_scale_x
            ramp[1] *= self.h_scale_y
            # apply homography
            ramp = ramp.reshape(2, -1)  # [2, H, W]
            H = self.homographies[index]
            [xt, yt] = transform2h(ramp[0], ramp[1], torch.inverse(H))
            # scale from world to [0,1]
            xt -= self.h_bounds_x[0]
            xt /= (self.h_bounds_x[1] - self.h_bounds_x[0])
            yt -= self.h_bounds_y[0]
            yt /= (self.h_bounds_y[1] - self.h_bounds_y[0])
            # restore shape
            ramp = torch.stack([xt.reshape(h, w), yt.reshape(h, w)], 0)
        return ramp

    def mask2trimap(self, mask):
        """Convert binary mask to trimap with values in [-1, 0, 1]."""
        fg_mask = (mask > 0).float()
        bg_mask = (mask < 0).float()
        trimap_width = getattr(self.opt, 'trimap_width', 20)
        trimap_width *= bg_mask.shape[-1] / self.opt.width
        trimap_width = int(trimap_width)
        bg_mask = cv2.erode(bg_mask.numpy(), kernel=np.ones((trimap_width, trimap_width)), iterations=1)
        bg_mask = torch.from_numpy(bg_mask)
        mask = fg_mask - bg_mask
        return mask
