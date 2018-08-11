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

from chainer import variable
from chainer import reporter
from chainer import initializers

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
    def __init__(self, in_channels, out_channels, ksize=3, pad=1):
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
        self.pc = Variable(xp.zeros((shape[0], self.out_channels, shape[2], shape[3]), dtype = xp.float32))
        self.ph = Variable(xp.zeros((shape[0], self.out_channels, shape[2], shape[3]), dtype = xp.float32))

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
        super(RepBlock, self).__init__()
        with self.init_scope():
            """
            convlstm : representation Convolution lstm.
            """
            self.convlstm = ConvLSTM(in_channels, out_channels)

    def reset_state(self):
        self.convlstm.reset_state()

    def __call__(self, before_E, before_R, top_R=None):
        """ 2 inputs(E(t-1,l), R(t-1,l+1)) & 1 output(R(t, l))
        before_E : E(t-1,l)
        top_R    : ConvLSTM's output from upper layer; R(t-1,l+1)
        """

        # Representation unit
        # Concat ConvLSTM inputs with axis = channel dim
        if top_R is not None:
            up_R = F.unpooling_2d(top_R, ksize=2, stride=2, cover_all=False)
            inputs_lstm = F.concat((before_R, before_E, up_R), axis=1)
            #print(inputs_lstm.shape)
        else:
            inputs_lstm = F.concat((before_R, before_E), axis=1)
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

    def __init__(self, out_channels, pixel_max=1.0, first_layer=False):
        super(ErrorBlock, self).__init__()
        with self.init_scope():
            """
            tconv : target Convolution.
            pconv : prediction Convolution.
            """
            if first_layer == False:
                self.tconv = L.Convolution2D(None, out_channels, ksize=3, pad=1)
            self.pconv = L.Convolution2D(None, out_channels, ksize=3, pad=1)
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
        E = F.concat((F.relu(Ahat-A), F.relu(A-Ahat)), axis=1)
        if self.first_layer == True:
            return E, Ahat
        else:
            return E

# Build Predictive Coding Network
class PredNet(chainer.Chain):
    def __init__(self, stack_sizes=(3, 48, 96, 192), return_Ahat=False,
                 prediction_length=0):
        """
        Args:
        """
        super(PredNet, self).__init__()
        with self.init_scope():
            num_layers = len(stack_sizes)
            for l in range(num_layers):
                if l != num_layers-1:
                    setattr(self, "R_block"+str(l),
                            RepBlock(stack_sizes[l]*3 + stack_sizes[l+1], stack_sizes[l]))
                else:
                    setattr(self, "R_block"+str(l),
                            RepBlock(stack_sizes[l]*3, stack_sizes[l]))

                if l == 0:
                    setattr(self, "E_block"+str(l),
                            ErrorBlock(stack_sizes[l], first_layer=True))
                else:
                    setattr(self, "E_block"+str(l), ErrorBlock(stack_sizes[l]))

            self.num_layers = num_layers
            self.stack_sizes = stack_sizes
            self.return_Ahat = return_Ahat
            if return_Ahat == True:
                self.prediction_length = prediction_length
            else:
                self.prediction_length = 0

    def reset_state(self, batch_size, im_height, im_width):
        # Set initial states
        h = [int(im_height*((0.5)**(i))) for i in range(self.num_layers)]
        w = [int(im_width*((0.5)**(i))) for i in range(self.num_layers)]
        e = [] # initial E
        r = [] # initial R
        for l in range(self.num_layers):
            e.append(Variable(xp.zeros((batch_size, self.stack_sizes[l]*2, h[l], w[l]), dtype=xp.float32)))
            r.append(Variable(xp.zeros((batch_size, self.stack_sizes[l], h[l], w[l]), dtype=xp.float32)))
            getattr(self, "R_block"+str(l)).reset_state()

        return e, r

    def __call__(self, x):
        """
        x.shape = (batch, time, channel, height, width)
        T : Video length
        """
        xs = x.shape
        T = xs[1]
        loss = None

        # Set initial states
        e_init, r_init = self.reset_state(xs[0], xs[3], xs[4])

        if self.return_Ahat == True:
            Ahat = []

        e_t = [None]*self.num_layers
        r_t = [None]*self.num_layers

        for t in range(T+self.prediction_length):
            if t == 0:
                e_tm1 = e_init # E(t minus 1)
                r_tm1 = r_init # R(t minus 1)
                ahat = None

            # Update R states
            for l in reversed(range(self.num_layers)):
                if l == self.num_layers-1:
                    r_t[l] = getattr(self, "R_block"+str(l))(e_tm1[l], r_tm1[l])
                else:
                    r_t[l] = getattr(self, "R_block"+str(l))(e_tm1[l], r_tm1[l], r_t[l+1])

            # Update Ahat, A, E states
            for l in range(self.num_layers):
                if l == 0:
                    if t < T:
                        e_t[l], frame_prediction = getattr(self, "E_block"+str(l))(x[:,t], r_t[l])
                    else:
                        e_t[l], frame_prediction = getattr(self, "E_block"+str(l))(ahat, r_t[l])    
                else:
                    e_t[l] = getattr(self, "E_block"+str(l))(e_t[l-1], r_t[l])

            if self.return_Ahat == True:
                ahat = frame_prediction
                Ahat.append(frame_prediction)

            if t > 0:
                loss_t = F.average(e_t[0])
            else:
                loss_t = 0
            loss = loss_t if loss is None else loss + loss_t

            # Update
            e_tm1 = e_t
            r_tm1 = r_t

        reporter.report({'loss': loss}, self)

        if self.return_Ahat == True:
            return loss, Ahat
        else:
            return loss
