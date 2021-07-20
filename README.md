# Omnimatte in PyTorch

This repository contains a re-implementation of the code for the CVPR 2021 paper "[Omnimatte: Associating Objects and Their Effects in Video](https://omnimatte.github.io/)."

<img src='./img/teaser.gif' height="260px"/>


## Prerequisites
- Linux
- Python 3.6+
- NVIDIA GPU + CUDA CuDNN

## Installation
This code has been tested with PyTorch 1.8 and Python 3.8.

- Install [PyTorch](http://pytorch.org) 1.8 and other dependencies.
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.
    
## Demo
To train a model on a video (e.g. "tennis"), run:
```bash
python train.py --name tennis --dataroot ./datasets/tennis --gpu_ids 0,1
```
To view training results and loss plots, visit the URL http://localhost:8097.
Intermediate results are also at `./checkpoints/tennis/web/index.html`.

To save the omnimatte layer outputs of the trained model, run:
```bash
python test.py --name tennis --dataroot ./datasets/tennis --gpu_ids 0
```
The results (RGBA layers, videos) will be saved to `./results/tennis/test_latest/`.

## Custom video
To train on your own video, you will have to preprocess the data:
1. Extract the frames, e.g.
    ```
    mkdir ./datasets/my_video && cd ./datasets/my_video 
    mkdir rgb && ffmpeg -i video.mp4 rgb/%04d.png
    ```
1. Resize the video to 256x448 and save the frames in `my_video/rgb`.
1. Get input object masks (e.g. using [Mask-RCNN](https://github.com/facebookresearch/detectron2) and [STM](https://github.com/seoungwugoh/STM)), save each object's masks in its own subdirectory, e.g. `my_video/mask/01/`, `my_video/mask/02/`, etc.
1. Compute flow (e.g. using [RAFT](https://github.com/princeton-vl/RAFT)), and save the forward .flo files to `my_video/flow` and backward flow to `my_video/flow_backward`
1. Compute the confidence maps from the forward/backward flows:
    ```bash
    python datasets/confidence.py --dataroot ./datasets/tennis
    ```
1. Register the video and save the computed homographies in `my_video/homographies.txt`.
See [here](docs/data.md#camera-registration) for details.

**Note**: Videos that are suitable for our method have the following attributes:
- Static camera or limited camera motion that can be represented with a homography.
- Limited number of omnimatte layers, due to GPU memory limitations. We tested up to 6 layers.
- Objects that move relative to the background (static objects will be absorbed into the background layer).
- We tested a video length of up to 200 frames (~7 seconds).

## Citation
If you use this code for your research, please cite the following paper:
```
@inproceedings{lu2021,
  title={Omnimatte: Associating Objects and Their Effects in Video},
  author={Lu, Erika and Cole, Forrester and Dekel, Tali and Zisserman, Andrew and Freeman, William T and Rubinstein, Michael},
  booktitle={CVPR},
  year={2021}
}
```

## Acknowledgments
This code is based on [retiming](https://github.com/google/retiming) and [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
