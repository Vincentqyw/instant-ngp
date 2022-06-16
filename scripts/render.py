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

sys.path.append('path_to_colmap_python_scripts')
import read_write_model
from read_write_model import Camera, Point3D,Image
from read_write_model import read_cameras_text, read_model, write_model,qvec2rotmat,rotmat2qvec

def load_cam_path(path):
    with open(path) as f:
        data = json.load(f)
    t = data["time"]
    frames = data["path"]
    return frames,t    

def load_transforms(path):
    print(f"path = {path}")
    with open(path) as f:
        frames = json.load(f)
    return frames

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

def colmap_to_nerf(xf, rot, totp, scale_trans):
    Tcw = np.copy(xf)
    Twc = np.linalg.inv(Tcw)
    
    T_r = np.eye(4)
    T_r[1,1] *= -1
    T_r[2,2] *= -1
    T_l = np.eye(4)
    T_l = T_l[[1,0,2,3],:]
    T_l[2,2] *= -1
    
    Twc = T_l @ Twc @ T_r
    
    # re-scale to center     
    Twc = rot @ Twc
    Twc[0:3,3] -= totp
    Twc[0:3,3] *= scale_trans
    
    return Twc


def nerf_to_colmap(xf, rot, totp, scale_trans):
    mat = np.copy(xf)
    mat[0:3,3] /= scale_trans
    mat[0:3,3] += totp
    
    R_inv = np.linalg.inv(rot)
    mat = R_inv @ mat
    
    T_r = np.eye(4)
    T_r[1,1] *= -1
    T_r[2,2] *= -1
    T_r_inv = np.linalg.inv(T_r)
    
    T_l = np.eye(4)
    T_l = T_l[[1,0,2,3],:]
    T_l[2,2] *= -1
    T_l_inv = np.linalg.inv(T_l)
    
    Twc = T_l_inv @ mat @ T_r_inv
    Tcw = np.linalg.inv(Twc)
    
    return Tcw


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
                 transform_path = "transform.json",
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

    # loading transforms -> colmap model files
    data = load_transforms(transform_path)
    rotation = np.array(data["rotation"])
    totp = np.array(data["totp"])
    sf = data["scale_trans"]
    images = {}

    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    xform = np.zeros([3,4]) # ngp pose
    
    testbed.fov_axis = 0
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

        if False:
            xform_nerf44 = np.concatenate([xform_nerf, bottom], 0)

            xform_colmap = nerf_to_colmap(xform_nerf44,rotation,totp,sf)

            qvec         = np.array(rotmat2qvec(xform_colmap[0:3,0:3]))
            tvec         = np.array(xform_colmap[0:3,3])
            camera_id    = int(_camera_id_)
            image_name   = f"{i:04d}.png"
            xys          = np.array([])
            point3D_ids  = np.array([])
            image_id     = i + 10000
            images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)

        # three ways to set camera render pose

        # 1. ngp -> nerf -> set_nerf_camera_matrix
        # testbed.set_nerf_camera_matrix(xform_nerf)

        # 2. ngp -> set_ngp_camera_matrix
        # testbed.set_ngp_camera_matrix(xform)

        # 3. set keyframe with additional params
        kf.fov = fov 
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
