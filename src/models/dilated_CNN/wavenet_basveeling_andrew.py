'''
Wavenet model from https://github.com/basveeling/wavenet adapted to
handle classification based on the methods described in the original wavenet paper
'''

from keras.layers import Input, Conv1D, Multiply, Add, Reshape, Activation, AveragePooling1D, Lambda, Flatten, Dense
from keras.models import load_model, Model
from keras.callbacks import History, ModelCheckpoint

import pandas as pd
import sys
import tensorflow as tf


class WaveNetClassifier:
    def __init__(self, input_shape, output_shape, kernel_size=2, dilation_depth=9, nb_stacks=1, nb_filters=40,
                 pool_size_1=80, pool_size_2=100, use_bias=False, use_skip_connections=False):
        """
        Parameters:
          input_shape: (tuple) tuple of input shape. (e.g. If input is 6s raw waveform with sampling rate = 16kHz, (96000,) is the input_shape)
          output_shape: (tuple)tuple of output shape. (e.g. If we want classify the signal into 100 classes, (100,) is the output_shape)
          kernel_size: (integer) kernel size of convolution operations in residual blocks
          dilation_depth: (integer) type total depth of residual blocks
          n_filters: (integer) # of filters of convolution operations in residual blocks
          load: (bool) load previous WaveNetClassifier or not
          load_dir: (string) the directory where the previous model exists
        """
        self.activation = 'softmax'
        self.scale_ratio = 1
        self.pool_size_1 = pool_size_1
        self.pool_size_2 = pool_size_2
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.dilation_depth = dilation_depth
        self.nb_filters = nb_filters
        self.use_bias = use_bias
        self.use_skip_connections = use_skip_connections
        self.input_shape = input_shape
        self.output_shape = output_shape

        if len(input_shape) == 1:
            self.expand_dims = True
        elif len(input_shape) == 2:
            self.expand_dims = False
        else:
            print('ERROR: wrong input shape')
            sys.exit()

        self.model = self.build_model()

    def residual_block(self, x, i, stack_nb):
        original_x = x
        tanh_out = Conv1D(self.nb_filters, 2, dilation_rate=2 ** i, padding='causal',
                          use_bias=self.use_bias,
                          name='dilated_conv_%d_tanh_s%d' % (2 ** i, stack_nb), activation='tanh')(x)
        sigm_out = Conv1D(self.nb_filters, 2, dilation_rate=2 ** i, padding='causal',
                          use_bias=self.use_bias,
                          name='dilated_conv_%d_sigm_s%d' % (2 ** i, stack_nb), activation='sigmoid')(x)
        x = Multiply(name='gated_activation_%d_s%d' % (i, stack_nb))([tanh_out, sigm_out])

        res_x = Conv1D(self.nb_filters, 1, padding='same', use_bias=self.use_bias)(x)
        skip_x = Conv1D(self.nb_filters, 1, padding='same', use_bias=self.use_bias)(x)
        res_x = Add()([original_x, res_x])
        return res_x, skip_x

    def build_model(self):
        input_layer = Input(shape=self.input_shape, name='input_part')
        out = input_layer
        skip_connections = []
        out = Conv1D(self.nb_filters, 2,
                     dilation_rate=1,
                     padding='causal',
                     name='initial_causal_conv'
                     )(out)
        for stack_nb in range(self.nb_stacks):
            for i in range(0, self.dilation_depth + 1):
                out, skip_out = self.residual_block(out, i, stack_nb)
                skip_connections.append(skip_out)

        if self.use_skip_connections:
            out = Add()(skip_connections)
        out = Activation('relu')(out)
        # added a mean-pooling layer after the dilated convolutions that aggregated the activations to coarser frames
        # spanning 10 milliseconds (160× downsampling)
        # mean pooling layer adjust pool_size_1 to change downsampling
        out = AveragePooling1D(self.pool_size_1, padding='same', name='mean_pooling_layer_downsampling')(out)


        # few non-causal convolutions
        # few non-causal convolutions
        out = Conv1D(self.nb_filters, self.pool_size_1, strides=2, padding='same', activation='relu')(out)
        out = Conv1D(self.nb_filters, self.pool_size_2, strides=2, padding='same', activation='relu')(out)
        out = Conv1D(self.output_shape, self.pool_size_2, strides=2, padding='same', activation='relu')(out)
        out = Conv1D(self.output_shape, self.pool_size_2, strides=2, padding='same', activation='relu')(out)


        out = Flatten()(out)
        out = Dense(512, activation='relu')(out)
        # add dropout?
        out = Dense(self.output_shape, activation='softmax')(out)

        return Model(input_layer, out)

    def get_model(self):
        return self.model

    def get_summary(self):
        self.model.summary()

    def get_receptive_field(self):
        return self.nb_stacks * (2 ** self.dilation_depth * 2) - (self.nb_stacks - 1)
