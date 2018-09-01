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

import network_extrap

def LoadData(file_name, num_frames=10):
    datalist = []

    all_file = np.load(file_name)

    count = all_file.shape[0]//num_frames
    #count = 10
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
    parser.add_argument('--opt', type=str, default=None)
    parser.add_argument('--epoch', '-e', type=int, default=60)
    parser.add_argument('--lr', '-l', type=float, default=0.001)
    parser.add_argument('--batch', '-b', type=int, default=4)
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    args = parser.parse_args()

    print("Loading datas")
    nt = 15
    extrap_start_time = 10  # starting at this time step, the prediction from the previous time step will be treated as the actual inp
    train = LoadData('X_train.npy', nt)
    validation = LoadData('X_val.npy', nt)

    # Set up a neural network to train.
    print("Building model")
    model = network_extrap.PredNet(stack_sizes=(3, 48, 96, 192), extrap_start_time=extrap_start_time)

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
    serializers.load_npz('./results/snapshot_iter_6898', trainer)

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
