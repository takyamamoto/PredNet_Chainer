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

def resize(image_dir="./2806/", out_dir="./data/", h=144, w=240):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    # Make the directory if it doesn't exist.
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    files_list = glob.glob(image_dir+"*.jpg")
    num_files = len(files_list)
    for i in tqdm(range(num_files)):
        #img = cv2.imread(files_list[i])
        img = cv2.imread(image_dir+"frame_{0:05d}_detection.jpg".format(1+i))
        img = cv2.resize(img, (w, h))
        cv2.imwrite(out_dir+"frame_{0:05d}.jpg".format(1+i),img)

if __name__ == '__main__':
    path = "./BikeVideoDataset"
    files = os.listdir(path)
    files_dir = [f for f in files if os.path.isdir(os.path.join(path, f))]
    for i in range(len(files_dir)):
        print("resize "+files_dir[i])
        resize(image_dir=path+"/"+files_dir[i]+"/", out_dir="./data/"+files_dir[i]+"_resized/")
