# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 04:19:32 2018

@author: user
"""

import argparse


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tqdm import tqdm

from chainer import cuda
from chainer import Variable
from chainer import serializers
import chainer.functions as F
import cv2

import network

xp = cuda.cupy

def LoadData(image_dir="./data/2806_resized/", begin=0, num_frames=10):
    datalist = []

    for j in range(num_frames):
        img = cv2.imread(image_dir+'frame_{0:05d}.jpg'.format(1+begin+j))
        img = xp.transpose(img, (2, 0, 1))
        img = xp.expand_dims(img, axis=0)
        if j == 0:
            stack = img
        else:
            stack = np.concatenate((stack, img), axis=0)
    stack = xp.expand_dims(stack, axis=0)
    datalist.append(stack)

    data = np.zeros((1, num_frames, 3, 144, 240))
    for i, partial_data in enumerate(tqdm(datalist)):
        data[i] = partial_data

    data = data.astype(xp.float32)
    data = data / 255
    assert data.max() == 1, "Data is not in range 0 to 1."

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--dir', '-d', type=str, default="./data/2806_resized/")
    parser.add_argument('--model', '-m', type=str, default="./results/model")
    parser.add_argument('--begin', '-b', type=int, default=0)
    args = parser.parse_args()

    # Set up a neural network to train.
    print("Loading data")
    num_frames = 10
    test = LoadData(image_dir=args.dir, begin=args.begin, num_frames=num_frames)
    x = test

    # Set up a neural network to train.
    print("Building model")
    prediction_length=5
    model = network.PredNet(return_Ahat=True, prediction_length=prediction_length)
    if args.model != None:
        print( "Loading model from " + args.model )
        serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        cuda.get_device_from_id(0).use()
        model.to_gpu()
        x = cuda.cupy.array(x)

    loss, predicted = model(Variable(x))
    print("Loss:", loss)
    print("len_outputs:", len(predicted))

    x = test * 255
    x = x.astype(np.uint8)
    x = np.reshape(x, (num_frames, 3, 144, 240))
    x = np.transpose(x, (0, 2, 3, 1))
    print(predicted[0].shape)

    #setup figure
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    ax1.set_title('Actual')
    ax2.set_title('Predicted')

    ims=[]
    for time in range(num_frames+prediction_length):
        title = fig.text(0.5, 0.85, "t = "+str(time+1), fontsize="large")
        if time < num_frames:
            x_rgb = cv2.cvtColor(x[time], cv2.COLOR_BGR2RGB)
            im, = [ax1.imshow(x_rgb)]
        else:
            im, = [ax1.imshow(np.zeros((144, 240, 3)))]

        p = cuda.to_cpu(predicted[time].data) #Variable to numpy
        p = np.reshape(p, (3, 144, 240))
        p = np.transpose(p, (1, 2, 0))
        p = p * 255
        p = p.astype(np.uint8)
        p = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
        im2, = [ax2.imshow(p)]
        ims.append([im, im2, title])

    #run animation
    ani = animation.ArtistAnimation(fig,ims, interval=500)

    plt.tight_layout()
    plt.show() #表示
    ani.save("results_video.mp4") #保存


if __name__ == '__main__':
    main()
