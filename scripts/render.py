#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os, sys, shutil
import argparse
from tqdm import tqdm

import common
import pyngp as ngp # noqa
import numpy as np

def render_video(resolution, numframes, scene, name, spp, fps, exposure=0):
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.load_snapshot(os.path.join(scene, "base.msgpack"))
    testbed.load_camera_path(os.path.join(scene, "base_cam.json"))

    tmp_dir = os.path.join(scene, "temp")

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    # if 'temp' in os.listdir():
        # shutil.rmtree('temp')

    for i in tqdm(list(range(min(numframes,numframes+1))), unit="frames", desc=f"Rendering"):
        testbed.camera_smoothing = i > 0
        frame = testbed.render(resolution[0], resolution[1], spp, True, float(i)/numframes, float(i + 1)/numframes, fps, shutter_fraction=0.5)
        tmp_path = os.path.join(scene, f"temp/{i:04d}.jpg")
        common.write_image(tmp_path, np.clip(frame * 2**exposure, 0.0, 1.0), quality=100)

    os.system(f"ffmpeg -i {tmp_dir}/%04d.jpg -vf \"fps={fps}\" -c:v libx264 -pix_fmt yuv420p {scene}/{name}_test.mp4")
    # shutil.rmtree('temp')


def parse_args():
    parser = argparse.ArgumentParser(description="render neural graphics primitives testbed, see documentation for how to")
    parser.add_argument("--scene", "--training_data", default="", help="The scene to load. Can be the scene's name or a full path to the training data.")

    parser.add_argument("--width", "--screenshot_w", type=int, default=1920, help="Resolution width of the render video")
    parser.add_argument("--height", "--screenshot_h", type=int, default=1080, help="Resolution height of the render video")
    parser.add_argument("--n_seconds", type=int, default=1, help="Number of steps to train for before quitting.")
    parser.add_argument("--fps", type=int, default=60, help="number of fps")
    parser.add_argument("--render_name", type=str, default="", help="name of the result video")


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()	

    render_video([args.width, args.height], args.n_seconds*args.fps, args.scene, args.render_name, spp=8, fps=args.fps)
