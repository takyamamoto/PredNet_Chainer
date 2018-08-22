# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 04:34:16 2018

@author: user
"""

import cv2
import numpy as np

W = 160
H = 128
bar_y = 35
bar_x = 30
bar_h = 10
bar_w = 50
x = 50
y = 50
h = 100
s = 5
for i in range(20):
    img = np.full((H, W, 3), 200, dtype=np.uint8)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (0,200,0), thickness=-1)
    cv2.rectangle(img, (W-bar_x-bar_w, bar_y), (W-bar_x, bar_y+bar_h), (0,200,0), thickness=-1)
    #cv2.rectangle(img, (x+s, y+s), (x+w+s, H-y+s), (0,0,0), thickness=-1)
    cv2.rectangle(img, (x, y), (W-x, y+h), (128,128,128), thickness=-1)
    cv2.rectangle(img, (x, y), (W-x, y+h), (0, 0, 0), thickness=1)
    cv2.imwrite('frames_{0:05d}.jpg'.format(i), img)
    y -= 6
    bar_y += 5