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

import network

xp = cuda.cupy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--model', '-m', type=str, default="./results/model")
    parser.add_argument('--begin', '-b', type=int, default=0)
    args = parser.parse_args()

    # Set up a neural network to train.
    print("Load data...")
    data = np.load("mnist_test_seq.npy")
    data = np.hstack((data[:10], data[10:]))
    data = data.transpose((1, 0, 2, 3))
    data = np.reshape(data, (20000, 10, 1, 64, 64))
    data = data / 255
    data = data.astype(np.float32)

    #train = data[:18000]
    #validation = data[18000:19000]
    test = data[19000:]
    which = 20

    # Set up a neural network to train.
    print("Build model...")
    model = network.PredNet(return_Ahat=True)
    if args.model != None:
        print( "Loading model from " + args.model )
        serializers.load_npz(args.model, model)

    x = test[which]
    x = np.expand_dims(x, 0)

    if args.gpu >= 0:
        cuda.get_device_from_id(0).use()
        model.to_gpu()
        x = cuda.cupy.array(x)

    loss, predicted = model(Variable(x))
    print("Loss:", loss)
    print("len_outputs:", len(predicted))

    x = test[which]
    x = x * 255
    x = x.astype(np.uint8)

    #setup figure
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    ax1.set_title('Actual')
    ax2.set_title('Predicted')

    ims=[]
    for time in range(10):
        title = fig.text(0.5, 0.85, "t = "+str(time), fontsize="large")
        im, = [ax1.imshow(x[time,0])]

        p = cuda.to_cpu(predicted[time].data) #Variable to numpy
        p = p * 255
        p = p.astype(np.uint8)
        im2, = [ax2.imshow(p[0,0])]
        ims.append([im, im2, title])

    #run animation
    ani = animation.ArtistAnimation(fig,ims, interval=500)

    plt.tight_layout()
    plt.show() #表示
    ani.save("results_video.mp4") #保存

    """
    #plt.savefig("result.png")
    #plt.show()
    #plt.close()
    y_predict = model.predict(x_test)
    y_subtraction = 1-np.abs(x_test - y_predict)
    # Display the 1st 8 corrupted and denoised images
    rows, cols = 1, 100
    num = rows * cols
    num_start = 5000
    imgs = np.concatenate([x_test[num_start:num_start+num], y_test[num_start:num_start+num],
                           y_predict[num_start:num_start+num], y_subtraction[num_start:num_start+num]])
    imgs = imgs.reshape((rows * 4, cols, 64, 64))
    imgs = np.vstack(np.split(imgs, rows, axis=1))
    imgs = imgs.reshape((rows * 4, -1, 64, 64))
    imgs = np.vstack([np.hstack(i) for i in imgs])
    imgs = (imgs * 255).astype(np.uint8)
    plt.figure()
    plt.axis('off')
    plt.imshow(imgs, interpolation='none', cmap='gray')
    Image.fromarray(imgs).save('AE_result3.png')
    plt.show()
    """

if __name__ == '__main__':
    main()
