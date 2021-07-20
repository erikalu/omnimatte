"""Script to save the full outputs of an Omnimatte model.

Once you have trained the Omnimatte model with train.py, you can use this script to save the model's final omnimattes.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates a model and dataset given the options. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (after training a model):
    python test.py --dataroot ./datasets/tennis --name tennis

    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

See options/base_options.py and options/test_options.py for more test options.
"""
import os
from options.test_options import TestOptions
from third_party.data import create_dataset
from third_party.models import create_model
from third_party.util.visualizer import save_images, save_videos
from third_party.util import html
import torch


if __name__ == '__main__':
    testopt = TestOptions()
    opt = testopt.parse()
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    video_visuals = None
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        visuals = model.get_results()  # rgba, reconstruction, original, mask
        if video_visuals is None:
            video_visuals = visuals
        else:
            for k in video_visuals:
                video_visuals[k] = torch.cat((video_visuals[k], visuals[k]))
        rgba = { k: visuals[k] for k in visuals if 'rgba' in k }
        # save RGBA layers
        save_images(webpage, rgba, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    save_videos(webpage, video_visuals, width=opt.display_winsize)
    webpage.save()  # save the HTML of videos
