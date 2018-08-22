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

import network_extrap

xp = cuda.cupy

def LoadData(file_name, num_frames=10):
    datalist = []
    all_file = np.load(file_name)
    count = all_file.shape[0]//num_frames
    for i in tqdm(range(count)):
        f = all_file[i*num_frames:(i+1)*num_frames]
        f = xp.transpose(f, (0, 3, 1, 2))
        datalist.append(f)

    data = np.zeros((count, num_frames, 3, 128, 160))
    for i, partial_data in enumerate(tqdm(datalist)):
        data[i] = partial_data

    print(data.shape)
    data = data.astype(xp.float32)
    data = data / 255
    assert data.max() == 1, "Data is not in range 0 to 1."

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--model', '-m', type=str, default="./results/model")
    parser.add_argument('--begin', '-b', type=int, default=0)
    args = parser.parse_args()

    # Set up a neural network to train.
    print("Loading data")
    nt = 15
    extrap_start_time = 10
    test = LoadData('X_test.npy', nt)
    which = 10
    x = test[which]
    x = np.expand_dims(x, 0)

    # Set up a neural network to train.
    print("Building model")  
    model = network_extrap.PredNet(return_Ahat=True, extrap_start_time=extrap_start_time)
    if args.model != None:
        print( "Loading model from " + args.model)
        serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        cuda.get_device_from_id(0).use()
        model.to_gpu()
        x = cuda.cupy.array(x)

    loss, predicted = model(Variable(x))
    print("Loss:", loss)
    print("len_outputs:", len(predicted))

    x = test[which] * 255
    x = x.astype(np.uint8)
    x = np.transpose(x, (0, 2, 3, 1))
    print(predicted[0].shape)

    #setup figure
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    ax1.set_title('Actual')
    ax2.set_title('Predicted')

    ims=[]
    for time in range(nt):
        title = fig.text(0.5, 0.85, "t = "+str(time+1), fontsize="large")
        if time < extrap_start_time:
            im, = [ax1.imshow(x[time])]
        else:
            im, = [ax1.imshow(np.zeros((128, 160, 3)))]

        p = cuda.to_cpu(predicted[time].data) #Variable to numpy
        p = np.reshape(p, (3, 128, 160))
        p = np.transpose(p, (1, 2, 0))
        p = p * 255
        p = p.astype(np.uint8)
        #p = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
        im2, = [ax2.imshow(p)]
        ims.append([im, im2, title])

    #run animation
    ani = animation.ArtistAnimation(fig,ims, interval=500)

    plt.tight_layout()
    plt.show() #表示
    ani.save("results_video.mp4") #保存


if __name__ == '__main__':
    main()
