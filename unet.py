# 3D model of unet

from tensorflow.python.keras.layers import Input, Dense, Convolution3D, Convolution2D, MaxPooling3D, MaxPooling2D, Dropout, Flatten, SpatialDropout3D, SpatialDropout2D, \
	Activation, AveragePooling3D,AveragePooling2D, UpSampling3D,UpSampling2D, BatchNormalization, ConvLSTM2D, \
	TimeDistributed, Concatenate, Lambda, Reshape
from tensorflow.python.keras.models import Model
import tensorflow as tf


def convolution_block_3(x, nr_of_convolutions, use_bn=False, spatial_dropout=None):
    for i in range(2):
        x = Convolution3D(nr_of_convolutions, 3, padding='same')(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if spatial_dropout:
            x = SpatialDropout3D(spatial_dropout)(x)

    return x

def encoder_block_3(x, nr_of_convolutions, use_bn=False, spatial_dropout=None):

    x_before_downsampling = convolution_block_3(x, nr_of_convolutions, use_bn, spatial_dropout)
    x = MaxPooling3D((2, 2, 2))(x_before_downsampling)

    return x, x_before_downsampling

def decoder_block_3(x, nr_of_convolutions, cross_over_connection=None, use_bn=False, spatial_dropout=None):

    x = UpSampling3D((2, 2, 2))(x)
    if cross_over_connection is not None:
        x = Concatenate()([cross_over_connection, x])
    x = convolution_block_3(x, nr_of_convolutions, use_bn, spatial_dropout)

    return x

#-------

def convolution_block_2(x, nr_of_convolutions, use_bn=False, spatial_dropout=None):
    for i in range(2):
        x = Convolution2D(nr_of_convolutions, 3, padding='same')(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if spatial_dropout:
            x = SpatialDropout2D(spatial_dropout)(x)

    return x

def encoder_block_2(x, nr_of_convolutions, use_bn=False, spatial_dropout=None):

    x_before_downsampling = convolution_block_2(x, nr_of_convolutions, use_bn, spatial_dropout)
    x = MaxPooling2D((2, 2))(x_before_downsampling)

    return x, x_before_downsampling

def decoder_block_2(x, nr_of_convolutions, cross_over_connection=None, use_bn=False, spatial_dropout=None):

    x = UpSampling2D((2, 2))(x)
    if cross_over_connection is not None:
        x = Concatenate()([cross_over_connection, x])
    x = convolution_block_2(x, nr_of_convolutions, use_bn, spatial_dropout)

    return x



class Unet():
	def __init__(self, input_shape, nb_classes):
		if len(input_shape) == 4:
			self.dim = 3
		elif len(input_shape) == 3:
			self.dim = 2
		else:
			raise('input shape must have either 3 or 4 dimesions')
		if nb_classes <= 1:
			raise ValueError('Segmentation classes must be > 1')
		self.input_shape = input_shape
		self.nb_classes = nb_classes
		self.convolutions = None
		self.encoder_use_bn = True
		self.decoder_use_bn = True
		self.encoder_spatial_dropout = None
		self.decoder_spatial_dropout = None
		print('Dimension of the network:', self.dim)
	def set_convolutions(self, convolutions):
		#if len(convolutions) != self.get_depth()*2 + 1:
		#	raise ValueError('Nr of convolutions must have length ' + str(self.get_depth()*2 + 1))
		self.convolutions = convolutions

	def get_depth(self):
		if self.dim == 3:
			init_size = min(self.input_shape[0], self.input_shape[1], self.input_shape[2])
		if self.dim == 2:
			init_size = min(self.input_shape[0], self.input_shape[1])
		size = init_size
		depth = 0
		while size % 2 == 0 and size > 4:
			size /= 2
			depth += 1

		return depth

	def get_dice_loss(self):
		def dice_loss(target, output, epsilon=1e-10):
			smooth = 1.
			dice = 0
			if self.dim == 3:
				for object in range(1, self.nb_classes):
					output1 = output[:, :, :, :, object]
					target1 = target[:, :, :, :, object]
					intersection1 = tf.reduce_sum(output1 * target1)
					union1 = tf.reduce_sum(output1 * output1) + tf.reduce_sum(target1 * target1)
					dice += (2. * intersection1 + smooth) / (union1 + smooth)
			if self.dim == 2:
				for object in range(1, self.nb_classes):
					output1 = output[:, :, :, object]
					target1 = target[:, :, :, object]
					intersection1 = tf.reduce_sum(output1 * target1)
					union1 = tf.reduce_sum(output1 * output1) + tf.reduce_sum(target1 * target1)
					dice += (2. * intersection1 + smooth) / (union1 + smooth)

			dice /= (self.nb_classes - 1)

			return tf.clip_by_value(1. - dice, 0., 1. - epsilon)

		return dice_loss

	def create(self):

		"""
		Create model and return it

		:return: keras model
		"""

		input_layer = Input(shape=self.input_shape)
		x = input_layer
		if self.dim == 3:
			init_size = min(self.input_shape[0], self.input_shape[1], self.input_shape[2])
		if self.dim == 2:
			init_size = min(self.input_shape[0], self.input_shape[1])			 
		size = init_size

		convolutions = self.convolutions
		if convolutions is None:
			# Create convolutions
			convolutions = []
			nr_of_convolutions = 8
			for i in range(self.get_depth()):
				convolutions.append(nr_of_convolutions)
				nr_of_convolutions *= 2
			convolutions.append(nr_of_convolutions)
			for i in range(self.get_depth()):
				convolutions.append(nr_of_convolutions)
				nr_of_convolutions /= 2

		if self.dim == 3:
			connection = {}
			i = 0
			while size % 2 == 0 and size > 4:
				x, connection[size] = encoder_block_3(x, convolutions[i], self.encoder_use_bn, self.encoder_spatial_dropout)
				size /= 2
				i += 1

			x = convolution_block_3(x, convolutions[i], self.encoder_use_bn, self.encoder_spatial_dropout)
			i += 1

			while size < init_size:
				size *= 2
				x = decoder_block_3(x, convolutions[i], connection[size], self.decoder_use_bn, self.decoder_spatial_dropout)
				i += 1


			x = Convolution3D(self.nb_classes, 1, activation='softmax')(x)
				
		if self.dim == 2:
			connection = {}
			i = 0
			while size % 2 == 0 and size > 4:
				x, connection[size] = encoder_block_2(x, convolutions[i], self.encoder_use_bn, self.encoder_spatial_dropout)
				size /= 2
				i += 1

			x = convolution_block_2(x, convolutions[i], self.encoder_use_bn, self.encoder_spatial_dropout)
			i += 1

			while size < init_size:
				size *= 2
				x = decoder_block_2(x, convolutions[i], connection[size], self.decoder_use_bn, self.decoder_spatial_dropout)
				i += 1


			x = Convolution2D(self.nb_classes, 1, activation='softmax')(x)


		return Model(inputs=input_layer, outputs=x)