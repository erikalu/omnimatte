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


"""Helper tools for computing the world bounds from homographies."""
import os
import sys
sys.path.append('.')
from utils import readFlow, numpy2im
import glob
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F


def transform2h(x, y, m):
    """Applies 2d homogeneous transformation."""
    A = np.dot(m, np.array([x, y, np.ones(len(x))]))
    xt = A[0, :] / A[2, :]
    yt = A[1, :] / A[2, :]
    return xt, yt


def compute_world_bounds(homographies, height, width):
    """Compute minimum and maximum coordinates.

    homographies - list of 3x3 numpy arrays
    height, width - video dimensions
    """
    xbounds = [0, width - 1]
    ybounds = [0, height - 1]

    for h in homographies: 
        # find transformed image bounding box
        x = np.array([0, width - 1, 0, width - 1])
        y = np.array([0, 0, height - 1, height - 1])
        [xt, yt] = transform2h(x, y, np.linalg.inv(h))
        xbounds[0] = min(xbounds[0], min(xt))
        xbounds[1] = max(xbounds[1], max(xt))
        ybounds[0] = min(ybounds[0], min(yt))
        ybounds[1] = max(ybounds[1], max(yt))

    return xbounds, ybounds


if __name__ == "__main__":
    import argparse
    arguments = argparse.ArgumentParser()
    arguments.add_argument('--homography_path', type=str, help='path to text file containing homographies')
    arguments.add_argument('--width', type=int, help='video width')
    arguments.add_argument('--height', type=int, help='video height')
    opt = arguments.parse_args()

    with open(opt.homography_path) as f:
        lines = f.readlines()
    homographies = [l.rstrip().split(' ') for l in lines]
    homographies = [[float(h) for h in l] for l in homographies]
    homographies = [np.array(H).reshape(3, 3) for H in homographies]
    xbounds, ybounds = compute_world_bounds(homographies, opt.height, opt.width)
    out_path = f'{opt.homography_path[:-4]}-final.txt'
    with open(out_path, 'w') as f:
        f.write(f'size: {opt.width} {opt.height}\n')
        f.write(f'bounds: {xbounds[0]} {xbounds[1]} {ybounds[0]} {ybounds[1]}\n')
        f.writelines(lines)
    print(f'saved {out_path}')
