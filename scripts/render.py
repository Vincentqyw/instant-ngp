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

import commentjson as json
from scipy.spatial.transform import Rotation as R


def load_cam_path(path):
    with open(path) as f:
        data = json.load(f)
    t = data["time"]
    frames = data["path"]
    return frames,t    

def ngp_to_nerf(xf):
    mat = np.copy(xf)
    mat = mat[[2,0,1],:] #swap axis

    mat[:,1] *= -1 #flip axis
    mat[:,2] *= -1

    mat[:,3] -= [0.5,0.5,0.5] # translation and re-scale
    mat[:,3] /= 0.33
    
    return mat

def nerf_to_ngp(xf):
    mat = np.copy(xf)
    mat = mat[:-1,:] 

    mat[:,1] *= -1 #flip axis
    mat[:,2] *= -1

    mat[:,3] *= 0.33
    mat[:,3] += [0.5,0.5,0.5] # translation and re-scale
    mat = mat[[1,2,0],:]
    return mat



def render_video(resolution, numframes, scene, name, spp, fps, 
                 snapshot = "base.msgpack",
                 cam_path = "base_cam.json",
                 exposure=0):
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.load_snapshot(os.path.join(scene, snapshot))
    testbed.load_camera_path(os.path.join(scene, cam_path))

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
    parser.add_argument("--snapshot", type=str, default="base.msgpack", help="name of nerf model")
    parser.add_argument("--cam_path", type=str, default="base_cam.json", help="name of the camera motion path")


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()	

    render_video([args.width, args.height], 
                 args.n_seconds*args.fps, 
                 args.scene, 
                 args.render_name, 
                 spp=8, 
                 snapshot = args.snapshot, 
                 cam_path = args.cam_path, 
                 fps=args.fps)
