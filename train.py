# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 08:51:18 2018

@author: user
"""

import argparse

import numpy as np
import cv2
import glob
from tqdm import tqdm

import chainer
from chainer import training
from chainer.training import extensions, triggers
from chainer import iterators, optimizers, serializers
from chainer import cuda
xp = cuda.cupy

import network

def LoadData(image_dir="./Denis_Day1_001/", num_frames=10, validation_rate=0.2):
    datalist = []
    num_files = len(glob.glob(image_dir+"*"))
    
    count = num_files//num_frames
    num_train = int(count*(1-validation_rate))
    
    for i in tqdm(range(count)):
        for j in range(num_frames):
            img = cv2.imread(image_dir+'frames_{0:06d}.jpg'.format(i*num_frames+j))
            img = xp.transpose(img, (2, 0, 1))
            img = xp.expand_dims(img, axis=0)
            if j == 0:
                stack = img
            else:
                stack = np.concatenate((stack, img), axis=0)
        stack = xp.expand_dims(stack, axis=0)
        datalist.append(stack)
    
    data = np.zeros((count, num_frames, 3, 120, 160))
    for i, partial_data in enumerate(tqdm(datalist)):
        data[i] = partial_data
    
    data = data.astype(xp.float32)
    data = data / 255
    assert data.max() == 1, "Data is not in range 0 to 1."
    
    return data[:num_train], data[num_train:]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--model', '-m', type=str, default=None)
    parser.add_argument('--opt', type=str, default=None)
    parser.add_argument('--epoch', '-e', type=int, default=60)
    parser.add_argument('--lr', '-l', type=float, default=0.001)
    parser.add_argument('--batch', '-b', type=int, default=8)
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    args = parser.parse_args()

    print("Loading data")
    
    # Denis_Day1_001,Munehiko_Day1_003,Michael_Day2_003,Alin_Day1_001,
    # Alin_Day1_002,Michael_Day2_002,Munehiko_Day1_002,Hussein_Day1_001
    path = '*.avi'
    file_list = glob.glob(path, recursive=True)
    name_list = [f[:-4] for f in file_list]
    train_list = []
    val_list = []
    N_train = 0
    N_var = 0
    num_frames = 20
    for i in range(4):
        print("Loading ", name_list[i])
        t, v = LoadData(image_dir="./"+name_list[i]+"/", num_frames=num_frames, validation_rate=0.2)
        train_list.append(t)
        val_list.append(v)
        N_train += t.shape[0]
        N_var += v.shape[0]

    train = np.zeros((N_train, num_frames, 3, 120, 160))
    validation = np.zeros((N_var, num_frames, 3, 120, 160))
    begin = 0
    for i, t in enumerate(tqdm(train_list)):
        train[begin:begin+t.shape[0]] = t
        begin += t.shape[0]
    
    begin = 0
    for i, v in enumerate(tqdm(val_list)):
        validation[begin:begin+v.shape[0]] = v
        begin += v.shape[0]
    #test = data[9500:]

    # Set up a neural network to train.
    print("Building model")
    model = network.PredNet(stack_sizes=(3, 48, 96, 192))

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = optimizers.Adam(alpha=args.lr)
    optimizer.setup(model)
    #optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(1e-4))

    train_iter = iterators.SerialIterator(train, batch_size=args.batch, shuffle=True)
    test_iter = iterators.SerialIterator(validation, batch_size=args.batch,
                                         repeat=False, shuffle=False)

    if args.model != None:
        print( "loading model from " + args.model)
        serializers.load_npz(args.model, model)

    if args.opt != None:
        print( "loading opt from " + args.opt)
        serializers.load_npz(args.opt, optimizer)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='results')

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.LogReport(trigger=(10, 'iteration')))
    #trainer.extend(extensions.observe_lr(observation_key='alpha'), trigger=(1, 'iteration'))
    #trainer.extend(extensions.observe_lr(), trigger=(10, 'iteration'))

    # Snapshot
    trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))
    #serializers.load_npz('./results/snapshot_iter_1407', trainer)

    # Decay learning rate
    points = [args.epoch*0.75]
    trainer.extend(extensions.ExponentialShift('alpha', 0.1),
                   trigger=triggers.ManualScheduleTrigger(points,'epoch'))


    # Save two plot images to the result dir
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))

    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']), trigger=(1, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=1))

    #Plot computation graph
    trainer.extend(extensions.dump_graph('main/loss'))

    # Train
    trainer.run()

    # Save results
    modelname = "./results/model"
    print( "saving model to " + modelname )
    serializers.save_npz(modelname, model)

    optimizername = "./results/optimizer"
    print( "saving optimizer to " + optimizername )
    serializers.save_npz(optimizername, optimizer)

if __name__ == '__main__':
    main()
