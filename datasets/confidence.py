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


"""Generate confidence maps from optical flow."""
import os
import sys
sys.path.append('.')
from utils import readFlow, numpy2im
import glob
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F


def compute_confidence(flo_f, flo_b, rgb, thresh=1, thresh_p=20):
    """Compute confidence map from optical flow."""
    im_height, im_width = flo_f.shape[:2]
    identity_grid = np.expand_dims(create_grid(im_height, im_width), 0)
    warp_b = flo_b[np.newaxis] + identity_grid
    warp_f = flo_f[np.newaxis] + identity_grid
    warp_b = map_coords(warp_b, im_height, im_width)
    warp_f = map_coords(warp_f, im_height, im_width)
    identity_grid = identity_grid.transpose(0, 3, 1, 2)
    warped_1 = F.grid_sample(torch.from_numpy(identity_grid), torch.from_numpy(warp_b), align_corners=True)
    warped_2 = F.grid_sample(warped_1, torch.from_numpy(warp_f), align_corners=True).numpy()
    err = np.linalg.norm(warped_2 - identity_grid, axis=1)
    err[err > thresh] = thresh
    err /= thresh
    confidence = 1 - err

    rgb = np.expand_dims(rgb.transpose(2, 0, 1), 0)
    rgb_warped_1 = F.grid_sample(torch.from_numpy(rgb).double(), torch.from_numpy(warp_b), align_corners=True)
    rgb_warped_2 = F.grid_sample(rgb_warped_1, torch.from_numpy(warp_f), align_corners=True).numpy()
    err = np.linalg.norm(rgb_warped_2 - rgb, axis=1)
    confidence_p = (err < thresh_p).astype(np.float32)
    confidence *= confidence_p

    return confidence[0]


def map_coords(coords, height, width):
    """Map coordinates from pixel-space to [-1, 1] range for torch's grid_sample function."""
    coords_mapped = np.stack([coords[..., 0] / (width - 1), coords[..., 1] / (height - 1)], -1)
    return coords_mapped * 2 - 1


def create_grid(height, width):
    ramp_u, ramp_v = np.meshgrid(np.linspace(0, width - 1, width), np.linspace(0, height - 1, height))
    return np.stack([ramp_u, ramp_v], -1)


if __name__ == "__main__":
    import argparse
    arguments = argparse.ArgumentParser()
    arguments.add_argument('--dataroot', type=str)
    opt = arguments.parse_args()

    forward_flo = sorted(glob.glob(os.path.join(opt.dataroot, 'flow', '*.flo')))
    backward_flo = sorted(glob.glob(os.path.join(opt.dataroot, 'flow_backward', '*.flo')))
    assert(len(forward_flo) == len(backward_flo))
    rgb_paths = sorted(glob.glob(os.path.join(opt.dataroot, 'rgb', '*')))
    print(f'generating {len(forward_flo)} confidence maps...')
    outdir = os.path.join(opt.dataroot, 'confidence')
    os.makedirs(outdir, exist_ok=True)
    for i in range(len(forward_flo)):
        flo_f = readFlow(forward_flo[i])
        flo_b = readFlow(backward_flo[i])
        rgb = np.array(Image.open(rgb_paths[i]))
        confidence = compute_confidence(flo_f, flo_b, rgb)
        fp = os.path.join(outdir, f'{i+1:04d}.png')
        im = numpy2im(confidence)
        im.save(fp)
    print(f'saved {len(forward_flo)} confidence maps to {outdir}')
