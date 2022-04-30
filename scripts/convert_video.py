import argparse
from email.mime import image
import os
from pathlib import Path, PurePosixPath

import numpy as np
import json
import sys
import math
import cv2
import os
import shutil

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Convert image into a different format. By default, converts to our binary fp16 '.bin' format, which helps quickly load large images.")
    parser.add_argument("--input", default="", help="Path to the image to convert.")
    parser.add_argument("--output", default="", help="Path to the output. Defaults to <input>.bin")
    parser.add_argument("--scale", default=4, help="scale")
    parser.add_argument("--show_image", default=0, help="scale")
    args = parser.parse_args()
    return args

def process_video(input,output,scale):

    cap = cv2.VideoCapture(input)

    cnt = 0
    # TODO: 
    interlace = 20

    while True:
        status,frame = cap.read()
        cnt += 1
        if cnt % interlace != 0:
            continue

        if status == False:
            break

        width,height = frame.shape[1],frame.shape[0]
        img_rescaled = cv2.resize(frame,(int(width/scale),int(height/scale)))

        if show_image == 1:
            cv2.imshow("image",img_rescaled)
            cv2.waitKey(10)
            # key = cv2.waitKey(1)
            output_dir = output + "/image_" + str(cnt) + ".png"
            cv2.imwrite(output_dir,img_rescaled)
        
if __name__ == "__main__":
    args = parse_args()
    scale       = int(args.scale)
    show_image  = int(args.show_image)
    # make dir
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    process_video(args.input,args.output,scale)
