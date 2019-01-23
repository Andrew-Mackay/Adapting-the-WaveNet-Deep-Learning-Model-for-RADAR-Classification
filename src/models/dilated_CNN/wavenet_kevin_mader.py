'''
This version of the wavenet based classifier was implemented by Kevin Mader
for the Kaggle "Quick, Draw!" Doodle Recognition Challenge and achieved a score of 0.63
https://www.kaggle.com/kmader/quickdraw-with-wavenet-classifier
'''
from keras.layers import Conv1D, Input, Activation, AveragePooling1D, Add, Multiply, GlobalAveragePooling1D
from keras.models import Model


class WaveNetClassifier:

    def __init__(self, input_shape, output_shape, n_filters=64, dilation_depth=8, activation='softmax',
                 scale_ratio=1, kernel_size=2, pool_size_1=4, pool_size_2=8):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_filters = n_filters
        self.dilation_depth = dilation_depth
        self.activation = activation
        self.scale_ratio = scale_ratio
        self.kernel_size = kernel_size
        self.pool_size_1 = pool_size_1
        self.pool_size_2 = pool_size_2
        self.model = self.make_model()

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

    def make_model(self):
        x = Input(shape=self.input_shape, name='original_input')
        skip_connections = []
        out = Conv1D(self.n_filters, 2, dilation_rate=1, padding='causal', name='dilated_conv_1')(x)
        for i in range(1, self.dilation_depth + 1):
            out, skip = self.residual_block(out, i)
            skip_connections.append(skip)
        out = Add(name='skip_connections')(skip_connections)
        out = Activation('relu')(out)

        out = Conv1D(self.n_filters, self.pool_size_1, strides=1, padding='same',
                     activation='relu', name='conv_5ms')(out)
        out = AveragePooling1D(self.pool_size_1, padding='same', name='downsample_to_200Hz')(out)

        out = Conv1D(self.n_filters, self.pool_size_2, padding='same',
                     activation='relu', name='conv_500ms')(out)
        out = Conv1D(self.output_shape[0], self.pool_size_2, padding='same',
                     activation='relu', name='conv_500ms_target_shape')(
            out)
        out = AveragePooling1D(self.pool_size_2, padding='same', name='downsample_to_2Hz')(out)
        out = Conv1D(self.output_shape[0], (int)(self.input_shape[0] / (self.pool_size_1 * self.pool_size_2)), padding='same',
                     name='final_conv')(out)
        out = GlobalAveragePooling1D(name='final_pooling')(out)
        out = Activation(self.activation, name='final_activation')(out)

        return Model(x, out)

    def get_model(self):
        return self.model

    def get_summary(self):
        self.model.summary()
