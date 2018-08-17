from tensorflow.python.keras.layers import Input, Dense, Convolution3D, MaxPooling3D, Dropout, Flatten, SpatialDropout3D, \
    ZeroPadding2D, Activation, AveragePooling2D, UpSampling2D, BatchNormalization, ConvLSTM2D, \
    TimeDistributed, Concatenate, Lambda, Reshape
from tensorflow.python.keras.models import Model
import tensorflow as tf


def convolution_block(x, nr_of_convolutions, use_bn=False, spatial_dropout=None):
    for i in range(2):
        x = Convolution3D(nr_of_convolutions, 3, padding='same')(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if spatial_dropout:
            x = SpatialDropout3D(spatial_dropout)(x)

    return x


def encoder_block(x, nr_of_convolutions, use_bn=False, spatial_dropout=None):

    x_before_downsampling = convolution_block(x, nr_of_convolutions, use_bn, spatial_dropout)
    x = MaxPooling3D((2, 2, 2))(x_before_downsampling)

    return x, x_before_downsampling


class VGGnet_3d():
    def __init__(self, input_shape, nb_classes):
        if len(input_shape) != 4:
            raise ValueError('Input shape must have 4 dimensions')
        if nb_classes <= 1:
            raise ValueError('Classes must be > 1')
        self.nb_classes = nb_classes
        self.input_shape = input_shape
        self.convolutions = None
        self.use_bn = True
        self.spatial_dropout = None
        self.dense_dropout = 0.5 #<----- default = 0.5 
        self.dense_size = 1024

    def set_dense_size(self, size):
        self.dense_size = size

    def set_dense_dropout(self, dropout):
        self.dense_dropout = dropout

    def set_spatial_dropout(self, dropout):
        self.spatial_dropout = dropout

    def set_convolutions(self, convolutions):
        if len(convolutions) != self.get_depth():
            raise ValueError('Nr of convolutions must have length ' + str(self.get_depth()*2 + 1))
        self.convolutions = convolutions

    def get_depth(self):
        init_size = min(self.input_shape[0], self.input_shape[1])
        size = init_size
        depth = 0
        while size % 2 == 0 and size > 4:
            size /= 2
            depth += 1

        return depth + 1

    def create(self):
        """
        Create model and return it

        :return: keras model
        """

        input_layer = Input(shape=self.input_shape)
        x = input_layer

        init_size = min(self.input_shape[0], self.input_shape[1], self.input_shape[2])
        size = init_size

        convolutions = self.convolutions
        if convolutions is None:
            # Create convolutions
            convolutions = []
            nr_of_convolutions = 8
            for i in range(self.get_depth()):
                convolutions.append(nr_of_convolutions)
                nr_of_convolutions *= 2

        i = 0
        while size % 2 == 0 and size > 4:
            x, _ = encoder_block(x, convolutions[i], self.use_bn, self.spatial_dropout)
            size /= 2
            i += 1

        x = convolution_block(x, convolutions[i], self.use_bn, self.spatial_dropout)

        x = Flatten(name="flatten")(x)
        x = Dense(self.dense_size, activation='relu')(x)
        x = Dropout(self.dense_dropout)(x)
        x = Dense(self.dense_size, activation='relu')(x)
        x = Dropout(self.dense_dropout)(x)
        x = Dense(self.nb_classes, activation='softmax')(x)

        return Model(inputs=input_layer, outputs=x)

