from keras.layers import Input, Conv1D, Multiply, Add, Reshape, Activation, AveragePooling1D, Lambda
from keras.models import load_model, Model
from keras.callbacks import History, ModelCheckpoint

import pandas as pd
import sys
import tensorflow as tf


# from https://github.com/mjpyeon/wavenet-classifier
class WaveNetClassifier:
    def __init__(self, input_shape, output_shape, kernel_size=2, dilation_depth=9, n_filters=40, pool_size_1=80,
                 pool_size_2=100, load=False, load_dir='./'):
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

        # save input info
        if len(input_shape) == 1:
            self.expand_dims = True
        elif len(input_shape) == 2:
            self.expand_dims = False
        else:
            print('ERROR: wrong input shape')
            sys.exit()
        self.input_shape = input_shape

        self.output_shape = output_shape

        # save hyperparameters of WaveNet
        self.kernel_size = kernel_size
        self.dilation_depth = dilation_depth
        self.n_filters = n_filters
        self.manual_loss = None

        if load is True:
            self.model = load_model(load_dir + "saved_wavenet_clasifier.h5", custom_objects={'tf': tf})
            self.prev_history = pd.read_csv(load_dir + 'wavenet_classifier_training_history.csv')
            self.start_idx = len(self.prev_history)
            self.history = None
        else:
            self.model = self.construct_model()
            self.start_idx = 0
            self.history = None
            self.prev_history = None

    def residual_block(self, x, i):
        tanh_out = Conv1D(self.n_filters,
                          self.kernel_size,
                          dilation_rate=self.kernel_size ** i,
                          padding='causal',
                          name='dilated_conv_%d_tanh' % (self.kernel_size ** i),
                          activation='tanh'
                          )(x)
        sigm_out = Conv1D(self.n_filters,
                          self.kernel_size,
                          dilation_rate=self.kernel_size ** i,
                          padding='causal',
                          name='dilated_conv_%d_sigm' % (self.kernel_size ** i),
                          activation='sigmoid'
                          )(x)
        z = Multiply(name='gated_activation_%d' % (i))([tanh_out, sigm_out])
        skip = Conv1D(self.n_filters, 1, name='skip_%d' % (i))(z)
        res = Add(name='residual_block_%d' % (i))([skip, x])
        return res, skip

    def construct_model(self):
        x = Input(shape=self.input_shape, name='original_input')
        if self.expand_dims:
            x_reshaped = Reshape(self.input_shape + (1,), name='reshaped_input')(x)
        else:
            x_reshaped = x
        skip_connections = []
        out = Conv1D(self.n_filters, 2, dilation_rate=1, padding='causal', name='dilated_conv_1')(x_reshaped)
        for i in range(1, self.dilation_depth + 1):
            out, skip = self.residual_block(out, i)
            skip_connections.append(skip)
        out = Add(name='skip_connections')(skip_connections)
        out = Activation('relu')(out)

        '''
        "For this task we added a mean-pooling layer after the dilated convolutions
        that agregated the activations to coarser frames spanning 10 milliseconds
        (160x downsampling). The pooling layer was followed by a few non-causal convolutions." - Wavenet Paper
        '''
        out = Conv1D(self.n_filters, self.pool_size_1, strides=1, padding='same', name='conv_5ms', activation='relu')(
            out)
        out = AveragePooling1D(self.pool_size_1, padding='same', name='downsample_to_200Hz')(out)

        out = Conv1D(self.n_filters, self.pool_size_2, padding='same', activation='relu', name='conv_500ms')(out)
        out = Conv1D(self.output_shape[0], self.pool_size_2, padding='same', activation='relu',
                     name='conv_500ms_target_shape')(out)
        out = AveragePooling1D(self.pool_size_2, padding='same', name='downsample_to_2Hz')(out)
        out = Conv1D(self.output_shape[0], (int)(self.input_shape[0] / (self.pool_size_1 * self.pool_size_2)),
                     padding='same', name='final_conv')(out)
        out = AveragePooling1D((int)(self.input_shape[0] / (self.pool_size_1 * self.pool_size_2)),
                               name='final_pooling')(out)
        out = Reshape(self.output_shape)(out)
        out = Activation(self.activation)(out)
        if self.scale_ratio != 1:
            out = Lambda(lambda x: x * self.scale_ratio, name='output_reshaped')(out)
        model = Model(x, out)
        return model

    def get_model(self):
        return self.model

    def add_loss(self, loss):
        self.manual_loss = loss

    def fit(self, X, Y, validation_data=None, epochs=100, batch_size=32, optimizer='adam', save=False, save_dir='./'):
        # set default losses if not defined
        if self.manual_loss is not None:
            loss = self.manual_loss
            metrics = None
        else:
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']

        # set callback functions
        if save:
            saved = save_dir + "saved_wavenet_clasifier.h5"
            hist = save_dir + 'wavenet_classifier_training_history.csv'
            if validation_data is None:
                checkpointer = ModelCheckpoint(filepath=saved, monitor='loss', verbose=1, save_best_only=True)
            else:
                checkpointer = ModelCheckpoint(filepath=saved, monitor='val_loss', verbose=1, save_best_only=True)
            history = History()
            callbacks = [history, checkpointer]
        else:
            callbacks = None

        # compile the model
        self.model.compile(optimizer, loss, metrics)
        try:
            self.history = self.model.fit(X, Y, shuffle=True, batch_size=batch_size, epochs=epochs,
                                          validation_data=validation_data, callbacks=callbacks,
                                          initial_epoch=self.start_idx)
        except:
            if save:
                df = pd.DataFrame.from_dict(history.history)
                df.to_csv(hist, encoding='utf-8', index=False)
            raise
            sys.exit()
        return self.history

    def predict(self, x):
        return self.model.predict(x)

    def get_summary(self):
        self.model.summary()
