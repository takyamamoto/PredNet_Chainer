# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 08:51:18 2018

@author: user
"""

import argparse

import chainer
from chainer import training
from chainer.training import extensions
from chainer import iterators, optimizers, serializers
from chainer import cuda
import chainer.links as L
from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100

import network

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--dataset', '-d', default='cifar10',
                        help='The dataset to use: cifar10 or cifar100')
    parser.add_argument('--model', '-m', type=str, default=None)
    parser.add_argument('--opt', type=str, default=None)
    parser.add_argument('--epoch', '-e', type=int, default=40)
    parser.add_argument('--looptimes', '-t', type=int, default=5)
    parser.add_argument('--lr', '-l', type=float, default=0.01)
    parser.add_argument('--batch', '-b', type=int, default=128)
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    args = parser.parse_args()

    if args.dataset == 'cifar10':
        print('Using CIFAR10 dataset.')
        class_labels = 10
        train, test = get_cifar10()
    elif args.dataset == 'cifar100':
        print('Using CIFAR100 dataset.')
        class_labels = 100
        train, test = get_cifar100()
    else:
        raise RuntimeError('Invalid dataset choice.')

    # Set up a neural network to train.
    model = L.Classifier(network.LocalPCN(class_labels=class_labels, LoopTimes=args.looptimes))

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = optimizers.NesterovAG(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(1e-4))

    num_train_samples = 45000
    train_iter = iterators.SerialIterator(train[:num_train_samples], batch_size=args.batch, shuffle=True)
    test_iter = iterators.SerialIterator(train[num_train_samples:], batch_size=args.batch, repeat=False, shuffle=False)

    if args.model != None:
        print( "loading model from " + args.model )
        serializers.load_npz(args.model, model)

    if args.opt != None:
        print( "loading opt from " + args.opt )
        serializers.load_npz(args.opt, optimizer)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='results')

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.LogReport(trigger=(10, 'iteration')))

    # Schedule of a learning rate (LinearShift)
    fifty = int(args.epoch * 0.5 * num_train_samples / args.batch)
    seventyfive = int(args.epoch * 0.75 * num_train_samples / args.batch)
    trainer.extend(extensions.LinearShift("lr", (args.lr, args.lr*0.1), (fifty,fifty+args.batch)))
    trainer.extend(extensions.LinearShift("lr", (args.lr*0.1, args.lr*0.01), (seventyfive,seventyfive+args.batch)))


    # Save two plot images to the result dir
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss','main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
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
