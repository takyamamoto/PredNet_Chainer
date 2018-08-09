# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 08:51:18 2018

@author: user
"""

import argparse

import numpy as np
import chainer
from chainer import training
from chainer.training import extensions, triggers
from chainer import iterators, optimizers, serializers
from chainer import cuda
#import matplotlib.pyplot as plt

xp = cuda.cupy

import network

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--model', '-m', type=str, default=None)
    parser.add_argument('--opt', type=str, default=None)
    parser.add_argument('--epoch', '-e', type=int, default=40)
    parser.add_argument('--lr', '-l', type=float, default=0.001)
    parser.add_argument('--batch', '-b', type=int, default=4)
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    args = parser.parse_args()

    print("Loading and preprocessing data")
    data = np.load("mnist_test_seq.npy")
    data = data.transpose((1, 0, 2, 3))
    data = np.reshape(data, (10000, 20, 1, 64, 64))
    data = data / 255
    data = data.astype(xp.float32)

    train = data[:9000]
    validation = data[9000:9900]
    #test = data[9900:]

    #plt.imshow(train[0,0,0])

    # Set up a neural network to train.
    print("Building model")
    model = network.PredNet()

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
        print( "Loading model from " + args.model)
        serializers.load_npz(args.model, model)

    if args.opt != None:
        print( "Loading opt from " + args.opt)
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
    print( "Saving model to " + modelname )
    serializers.save_npz(modelname, model)

    optimizername = "./results/optimizer"
    print( "Saving optimizer to " + optimizername )
    serializers.save_npz(optimizername, optimizer)

if __name__ == '__main__':
    main()
