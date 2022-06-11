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
    # mat[:,3] -= 0.025
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
    # mat[:,3] += 0.025

    return mat

def nerf_to_colmap(xf):
    mat = np.copy(xf)
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
        
        tmp_path = f"{tmp_dir}/{i:04d}.jpg"
        common.write_image(tmp_path, np.clip(frame * 2**exposure, 0.0, 1.0), quality=100)

    os.system(f"ffmpeg -i {tmp_dir}/%04d.jpg -vf \"fps={fps}\" -c:v libx264 -pix_fmt yuv420p {scene}/{name}_test.mp4")
    # shutil.rmtree('temp')

# test render image given base_cam.json file without smooth and spline
def render_frames(resolution, numframes, scene, name, 
                 spp = 1, 
                 fps = 24, 
                 fov = 50.625,
                 snapshot = "base.msgpack",
                 cam_path = "base_cam.json",
                 exposure=0):

    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.load_snapshot(os.path.join(scene, snapshot))
    testbed.shall_train = False

    # create a dir to save frames
    tmp_dir = os.path.join(scene, "temp_frames")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    cam_path = os.path.join(scene, cam_path)
    ngp_frames, t = load_cam_path(cam_path)


    xform = np.zeros([3,4]) # ngp pose
    
    testbed.fov = fov # todo: fix fov
    counter = 0
    for frame in tqdm(ngp_frames):

        qvec = frame['R']
        tvec = frame['t']
        mat  = R.from_quat(qvec).as_matrix()
        xform[:3,:3] = mat
        xform[:,-1:] = np.array(tvec).reshape(3,-1)

        # xform_nerf = ngp_to_nerf(xform)
        # testbed.set_nerf_camera_matrix(xform_nerf)

        testbed.set_ngp_camera_matrix(xform)

        testbed.render(resolution[0],resolution[1],spp)
        tmp_path = f"{tmp_dir}/{counter:04d}.jpg"
        common.write_image(tmp_path, np.clip(frame * 2**exposure, 0.0, 1.0), quality=100)
        counter += 1

    os.system(f"ffmpeg -i {tmp_dir}/%04d.jpg -vf \"fps={fps}\" -c:v libx264 -pix_fmt yuv420p {scene}/{name}_test.mp4")


# test render image given base_cam.json file with given time, result in smoothing path
def render_frames_spline(resolution, numframes, scene, name, 
                 spp = 1, 
                 fps = 24, 
                 fov = 50.625,
                 snapshot = "base.msgpack",
                 cam_path = "base_cam.json",
                 exposure=0):

    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.load_snapshot(os.path.join(scene, snapshot))
    testbed.shall_train = False

    cam_path = os.path.join(scene, cam_path)
    testbed.load_camera_path(cam_path)

    # create a dir to save frames
    tmp_dir = os.path.join(scene, "temp_frames")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    xform = np.zeros([3,4]) # ngp pose
    
    testbed.fov = fov # todo: fix fov
    for i in tqdm(list(range(min(numframes,numframes+1))), unit="frames", desc=f"Rendering"):
        testbed.camera_smoothing = i > 0
        ts = float(i)/numframes
        
        kf = testbed.get_camera_from_time(ts)
        
        # parse pose
        qvec = kf.R
        tvec = kf.T
        mat  = R.from_quat(qvec).as_matrix()
        xform[:3,:3] = mat
        xform[:,-1:] = np.array(tvec).reshape(3,-1)
        xform_nerf = ngp_to_nerf(xform)

        # three ways to set camera render pose

        # 1. ngp -> nerf -> set_nerf_camera_matrix
        # testbed.set_nerf_camera_matrix(xform_nerf)

        # 2. ngp -> set_ngp_camera_matrix
        # testbed.set_ngp_camera_matrix(xform)

        # 3. set keyframe with additional params
        testbed.set_camera_from_keyframe(kf)

        frame = testbed.render(resolution[0], resolution[1], spp, True)
        
        tmp_path = f"{tmp_dir}/{i:04d}.jpg"
        common.write_image(tmp_path, np.clip(frame * 2**exposure, 0.0, 1.0), quality=100)

    os.system(f"ffmpeg -i {tmp_dir}/%04d.jpg -vf \"fps={fps}\" -c:v libx264 -pix_fmt yuv420p {scene}/{name}_test.mp4")



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
