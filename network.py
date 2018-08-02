# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 09:13:59 2018

@author: user
"""

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer import cuda

xp = cuda.cupy

# http://www-waka.ist.osaka-u.ac.jp/brainJournal/lib/exe/fetch.php?media=%E3%83%9A%E3%83%BC%E3%82%B8:2017:170301_f-tomita.pdf
# ConvLSTM from https://github.com/joisino/ConvLSTM/blob/master/network.py
class ConvLSTM(chainer.Chain):
    def __init__(self, inp = 256, mid = 128, sz = 3):
        super(ConvLSTM, self).__init__(
            Wxi = L.Convolution2D(inp, mid, sz, pad = sz//2),
            Whi = L.Convolution2D(mid, mid, sz, pad = sz//2, nobias = True),
            Wxf = L.Convolution2D(inp, mid, sz, pad = sz//2),
            Whf = L.Convolution2D(mid, mid, sz, pad = sz//2, nobias = True),
            Wxc = L.Convolution2D(inp, mid, sz, pad = sz//2),
            Whc = L.Convolution2D(mid, mid, sz, pad = sz//2, nobias = True),
            Wxo = L.Convolution2D(inp, mid, sz, pad = sz//2),
            Who = L.Convolution2D(mid, mid, sz, pad = sz//2, nobias = True)
        )

        self.inp = inp
        self.mid = mid

        self.pc = None
        self.ph = None

        with self.init_scope():
            Wci_initializer = initializers.Zero()
            self.Wci = variable.Parameter(Wci_initializer)
            Wcf_initializer = initializers.Zero()
            self.Wcf = variable.Parameter(Wcf_initializer)
            Wco_initializer = initializers.Zero()
            self.Wco = variable.Parameter(Wco_initializer)

    def reset_state(self, pc = None, ph = None):
        self.pc = pc
        self.ph = ph

    def initialize_params(self, shape):
        self.Wci.initialize((self.mid, shape[2], shape[3]))
        self.Wcf.initialize((self.mid, shape[2], shape[3]))
        self.Wco.initialize((self.mid, shape[2], shape[3]))

    def initialize_state(self, shape):
        self.pc = Variable(self.xp.zeros((shape[0], self.mid, shape[2], shape[3]), dtype = self.xp.float32))
        self.ph = Variable(self.xp.zeros((shape[0], self.mid, shape[2], shape[3]), dtype = self.xp.float32))

    def __call__(self, x):
        if self.Wci.data is None:
            self.initialize_params(x.data.shape)

        if self.pc is None:
            self.initialize_state(x.data.shape)

        ci = F.hard_sigmoid(self.Wxi(x) + self.Whi(self.ph) + F.scale(self.pc, self.Wci, 1))
        cf = F.hard_sigmoid(self.Wxf(x) + self.Whf(self.ph) + F.scale(self.pc, self.Wcf, 1))
        cc = cf * self.pc + ci * F.tanh(self.Wxc(x) + self.Whc(self.ph))
        co = F.hard_sigmoid(self.Wxo(x) + self.Who(self.ph) + F.scale(cc, self.Wco, 1))
        ch = co * F.tanh(cc)

        self.pc = cc
        self.ph = ch

        return ch

# Representation unit
class RepBlock(chainer.Chain):
    """
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        ksize (int): The size of the filter in ffconv & fbconv is ksize x ksize.
        pad (int): The padding to use for the convolution.
    """

    def __init__(self, in_channels, out_channels, pixel_max=1, first_layer = False):
        super(Block, self).__init__()
        with self.init_scope():
            """
            convlstm : representation Convolution lstm.
            """
            self.convlstm = ConvLSTM(n, sz[0], 5)
            self.first_layer = first_layer

    def __call__(self, before_E, before_R, top_R):
        """ 3 inputs(E(t-1,l), R(t-1,l), R(t-1,l+1)) & 1 output(R(t, l))
        before_E : E(t-1,l)
        before_R : R(t-1,l)
        top_R    : ConvLSTM's output from upper layer; R(t-1,l+1)
        """

        # Representation unit
        # In Last layer, top_R = 0
        # Concat ConvLSTM inputs with axis = channel dim
        up_R = F.unpooling_2d(top_R, ksize=2, stride=2, cover_all=False)
        inputs_lstm = F.concat((before_E, up_R), axis=1)
        R = self.convlstm(inputs_lstm)

        return R

# Target & Prediction & Error units
class PredictionBlock(chainer.Chain):
    """
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        ksize (int): The size of the filter in ffconv & fbconv is ksize x ksize.
        pad (int): The padding to use for the convolution.
    """

    def __init__(self, in_channels, out_channels, pixel_max=1, first_layer = False):
        super(Block, self).__init__()
        with self.init_scope():
            """
            tconv : target Convolution.
            pconv : prediction Convolution.
            """
            self.bn = L.BatchNormalization(in_channels)
            self.tconv = L.Convolution2D(in_channels, out_channels, ksize=3, pad=1)
            self.pconv = L.Convolution2D(in_channels, out_channels, ksize=3, pad=1)
            self.first_layer = first_layer
            self.pixel_max = pixel_max

    def __call__(self, bottom_E, lateral_R):
        """
        bottom_E  : Error from lower layer; E(t,l-1)
        lateral_R : ConvLSTM's output from lateral layer; R(t,l)
        """
        # Target unit
        if self.first_layer == True:
            A = bottom_E
        else:
            A = F.relu(self.tconv(bottom_E))
            A = F.max_pooling_2d(A, ksize=2, stride=2)

        # Prediction unit
        if self.first_layer == True:
            Ahat = F.clipped_relu(self.pconv(lateral_R), z=self.pixel_max)
        else:
            Ahat = F.relu(self.pconv(lateral_R))

        # Error unit
        E = F.concat((F.relu(A-Ahat), F.relu(Ahat-A)), axis=1)

        return E

# Define Local Predictive Coding Network
class PredNet(chainer.Chain):
    def __init__(self, num_layer = 3):
        super(LocalPCN, self).__init__()
        with self.init_scope():
            for l in range(num_layer):
                block[l] = Block(3, 64, LoopTimes=LoopTimes)
            bn = L.BatchNormalization(512),
            fc = L.Linear(512, class_labels, nobias=True)

    def __call__(self, x):
        T = x.shape[0]
        c = []
        r = []
        e = []
        e.append(K.concatenate((e_up, e_down), axis=self.channel_axis))
        for t in range(T):
            if t == 0:
                self.block.reset_representation_state(0)
                self.block.reset_error_state(0)
            # Update R states

            r = self.Block.reset_state() # convlstm

            # Update hat_A, A, E states
            h = self.block1(x)
            h = self.block2(h)
            h = self.block3(h)
            h = F.max_pooling_2d(h, ksize=2, stride=2)
            h = self.block4(h)
            h = self.block5(h)
            h = F.max_pooling_2d(h, ksize=2, stride=2)
            h = self.block6(h)
            h = self.block7(h)
            h = self.block8(h)
            h = F.average(h, axis=(2,3)) # Global Average Pooling
            h = self.bn(h)
        return self.fc(h)
