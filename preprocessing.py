# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 09:52:09 2018

@author: user
"""

import os
import shutil
import cv2
from tqdm import tqdm
import glob

def Video2Frames(video_file ="Denis_Day1_001.avi", image_file='frames_%s.jpg',
                 width = 160, height = 120):
    
    image_dir = "./"+video_file[:-4]+"/"
    # Delete the entire directory tree if it exists.
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)

    # Make the directory if it doesn't exist.
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # Video to frames
    i = 0
    cap = cv2.VideoCapture(video_file)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    pbar = tqdm(total=num_frames)
    while(cap.isOpened()):
        flag, frame = cap.read()  # Capture frame-by-frame
        if flag == False:  # Is a frame left?
            break
        frame = cv2.resize(frame, (width, height))
        cv2.imwrite(image_dir+image_file % str(i).zfill(6), frame)  # Save a frame
        i += 1
        pbar.update(1)

    cap.release()  # When everything done, release the capture
    pbar.close()

if __name__ == '__main__':
    path = '*.avi'
    file_list = glob.glob(path, recursive=True)
    for file in file_list:
        print("Preprocessing ", file)
        Video2Frames(video_file=file)
