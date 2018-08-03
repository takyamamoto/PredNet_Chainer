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

# ConvLSTM from https://github.com/joisino/ConvLSTM/blob/master/network.py
class ConvLSTM(chainer.Chain):
    """
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        ksize (int): The size of the filter.
        pad (int): The padding to use for the convolution.
    """
    def __init__(self,in_channels, out_channels, ksize=3, pad=1):
        super(ConvLSTM, self).__init__(
            Wxi = L.Convolution2D(in_channels, out_channels, ksize=ksize, pad=pad),
            Whi = L.Convolution2D(out_channels, out_channels, ksize=ksize, pad=pad, nobias = True),
            Wxf = L.Convolution2D(in_channels, out_channels, ksize=ksize, pad=pad),
            Whf = L.Convolution2D(out_channels, out_channels, ksize=ksize, pad=pad, nobias = True),
            Wxc = L.Convolution2D(in_channels, out_channels, ksize=ksize, pad=pad),
            Whc = L.Convolution2D(out_channels, out_channels, ksize=ksize, pad=pad, nobias = True),
            Wxo = L.Convolution2D(in_channels, out_channels, ksize=ksize, pad=pad),
            Who = L.Convolution2D(out_channels, out_channels, ksize=ksize, pad=pad, nobias = True)
        )

        self.in_channels = in_channels
        self.out_channels = out_channels

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
        self.Wci.initialize((self.out_channels, shape[2], shape[3]))
        self.Wcf.initialize((self.out_channels, shape[2], shape[3]))
        self.Wco.initialize((self.out_channels, shape[2], shape[3]))

    def initialize_state(self, shape):
        self.pc = Variable(self.xp.zeros((shape[0], self.out_channels, shape[2], shape[3]), dtype = self.xp.float32))
        self.ph = Variable(self.xp.zeros((shape[0], self.out_channels, shape[2], shape[3]), dtype = self.xp.float32))

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

    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        with self.init_scope():
            """
            convlstm : representation Convolution lstm.
            """
            self.convlstm = ConvLSTM(in_channels, out_channels)

    def reset_state(self):
        self.convlstm.reset_state()

    def __call__(self, before_E, top_R=None):
        """ 2 inputs(E(t-1,l), R(t-1,l+1)) & 1 output(R(t, l))
        before_E : E(t-1,l)
        top_R    : ConvLSTM's output from upper layer; R(t-1,l+1)
        """

        # Representation unit
        # Concat ConvLSTM inputs with axis = channel dim
        if top_R is not None:
            up_R = F.unpooling_2d(top_R, ksize=2, stride=2, cover_all=False)
            inputs_lstm = F.concat((before_E, up_R), axis=1)
        else:
            inputs_lstm = before_E
        R = self.convlstm(inputs_lstm)
        return R

# Target & Prediction & Error units
class ErrorBlock(chainer.Chain):
    """
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        pixel_max: the maximum pixel value. Used to clip the pixel-layer prediction.
    """

    def __init__(self, t_in_channels, p_in_channels, out_channels, pixel_max=1, first_layer=False):
        super(Block, self).__init__()
        with self.init_scope():
            """
            tconv : target Convolution.
            pconv : prediction Convolution.
            """
            if first_layer == False:
                self.tconv = L.Convolution2D(t_in_channels, out_channels, ksize=3, pad=1)
            self.pconv = L.Convolution2D(p_in_channels, out_channels, ksize=3, pad=1)
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
            # F.clipped_relu equals SatLU + ReLU
            Ahat = F.clipped_relu(self.pconv(lateral_R), z=self.pixel_max)
        else:
            Ahat = F.relu(self.pconv(lateral_R))

        # Error unit
        E = F.concat((F.relu(A-Ahat), F.relu(Ahat-A)), axis=1)
        if self.first_layer == True:
            return E, Ahat
        else:
            return E

# Build Predictive Coding Network
class PredNet(chainer.Chain):
    def __init__(self, return_Ahat=False):
        super(LocalPCN, self).__init__()
        with self.init_scope():
            self.R_block1 = RepBlock(34, 1)
            self.R_block2 = RepBlock(128, 32)
            self.R_block3 = RepBlock(256, 64)
            self.R_block4 = RepBlock(256, 128)
            #self.R_block5 = RepBlock(1024, 256)

            self.E_block1 = ErrorBlock(1, 1, 1, first_layer=True)
            self.E_block2 = ErrorBlock(2, 32, 32)
            self.E_block3 = ErrorBlock(64, 64, 64)
            self.E_block4 = ErrorBlock(128, 128, 128)
            #self.E_block5 = ErrorBlock(256, 256, 256)

            self.return_Ahat = return_Ahat

    def __call__(self, x):
        """
        x.shape = (batch, time, channel, height, width)
        T : Video length
        """
        xs = x.shape
        T = xs[1]
        loss = None

        # Set initial states
        size = [int(xs[3]*((0.5)**(i))) for i in range(4)]
        init_e1 = Variable(self.xp.zeros((xs[0], 2, size[0], size[0]), dtype=self.xp.float32))
        init_e2 = Variable(self.xp.zeros((xs[0], 64, size[1], size[1]), dtype=self.xp.float32))
        init_e3 = Variable(self.xp.zeros((xs[0], 128, size[2], size[2]), dtype=self.xp.float32))
        init_e4 = Variable(self.xp.zeros((xs[0], 256, size[3], size[3]), dtype=self.xp.float32))

        e = [init_e1, init_e2, init_e3, init_e4]
        self.R_block1.reset_state()
        self.R_block2.reset_state()
        self.R_block3.reset_state()
        self.R_block4.reset_state()

        Ahat = []
        for t in range(T):
            # Update R states
            r4 = self.R_block4(e[3])
            r3 = self.R_block3(e[2], r4)
            r2 = self.R_block2(e[1], r3)
            r1 = self.R_block1(e[0], r2)
            r = [r1, r2, r3, r4]

            # Update Ahat, A, E states
            e1, ahat1 = self.E_block1(x[:,t], r[0])
            e2 = self.E_block1(e1, r[1])
            e3 = self.E_block1(e2, r[2])
            e4 = self.E_block1(e3, r[3])
            e = [e1, e2, e3, e4]

            Ahat.append(ahat1)

            if t > 0:
                loss_t = F.sum(e1)
                loss = loss_t if loss is None else loss + loss_t

        if self.return_Ahat == True:
            return Ahat, loss
        else:
            return loss
