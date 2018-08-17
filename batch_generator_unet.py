#batch gen
from smistad.smistad_dataset import get_dataset_files
import random
import h5py
import numpy as np
from scipy.ndimage.interpolation import rotate, shift, affine_transform, zoom
from numpy.random import random_sample, rand, random_integers, uniform
import matplotlib.pyplot as plt
from image_functions import maxminscale_bin


# quite slow
def add_affine_transform3(input_im, output,max_deform):

	random_20 = uniform(-max_deform,max_deform,2)
	random_80 = uniform(1-max_deform,1+max_deform,2)

	mat = np.array([[1, 0, 0],
					[0, random_80[0], random_20[0]],
					[0, random_20[1], random_80[1]]]
					)
	input_im[:, :, :, 0] = affine_transform(input_im[:, :, :,0], mat, output_shape = np.shape(input_im[:, :, :,0]))
	output[:, :, :, 0] = affine_transform(output[:, :, :,0], mat, output_shape = np.shape(input_im[:, :, :,0]))
	output[:, :, :, 1] = affine_transform(output[:, :, :,1], mat, output_shape = np.shape(input_im[:, :, :,0]))
	
	output[output < 0.5] = 0
	output[output >= 0.5] = 1
	
	return input_im, output

"""
###
input_im:		input image, 5d ex: (1,64,256,256,1) , (dimi0, z, x, y, chanell)
output:			ground truth, 5d ex: (1,64,256,256,2), (dimi0, z, x, y, chanell)
max_shift:		the maximum amount th shift in a direction, only shifts in x and y dir
###
"""
# quite slow
def add_shift3(input_im, output, max_shift):
	sequence = [0,
				random_sample()*max_shift - random_sample()*max_shift,
				random_sample()*max_shift - random_sample()*max_shift
	] 

	input_im[:, :, :, 0] = shift(input_im[:, :, :, 0], sequence)
	output[:, :, :, 0] = shift(output[:, :, :, 0], sequence)
	output[:, :, :, 1] = shift(output[:, :, :, 1], sequence)

	output[output < 0.5] = 0
	output[output >= 0.5] = 1

	return input_im, output

"""
####
input_im:		input image, 5d ex: (1,64,256,256,1) , (dimi0, z, x, y, chanell)
output:			ground truth, 5d ex: (1,64,256,256,2), (dimi0, z, x, y, chanell)
min/max_angle: 	minimum and maximum angle to rotate in deg, positive integers/floats.
####
"""	
def add_rotation3(input_im, output, max_angle):
	angle_xy = (random_sample() * max_angle) - (random_sample() * max_angle)
	angle_xz = (random_sample() * max_angle) - (random_sample() * max_angle)

	# rortate chunks
	input_im[:, :, :, 0] = rotate(input_im[:, :, :, 0], angle_xy, axes = (1,2), reshape = False, mode = 'nearest', order = 1)
	input_im[:, :, :, 0] = rotate(input_im[:, :, :, 0], angle_xz, axes = (0,1), reshape = False, mode = 'nearest', order = 1)
	
	output[:, :, :, 0] = rotate(output[:, :, :, 0], angle_xy, axes = (1,2), reshape = False, mode = 'nearest', order = 0)
	output[:, :, :, 0] = rotate(output[:, :, :, 0], angle_xz, axes = (0,1), reshape = False, mode = 'nearest', order = 0)
	output[:, :, :, 1] = rotate(output[:, :, :, 1], angle_xy, axes = (1,2), reshape = False, mode = 'nearest', order = 0)
	output[:, :, :, 1] = rotate(output[:, :, :, 1], angle_xz, axes = (0,1), reshape = False, mode = 'nearest', order = 0)
	
	return input_im, output


"""

"""
def add_flip3(input_im, output):
	flip_ax = random_integers(0, high = 2)

	# rortate chunks
	input_im[:, :, :, 0] = np.flip(input_im[:, :, :, 0], flip_ax)
	
	output[:, :, :, 0] = np.flip(output[:, :, :, 0], flip_ax)
	output[:, :, :, 1] = np.flip(output[:, :, :, 1], flip_ax)
	
	return input_im, output



"""
aug: 		dict with what augmentation as key and what degree of augmentation as value
		->  'rotate': 20 , in deg. slow
		->	'shift': 20, in pixels. slow
		->	'affine': 0.2 . shuld be between 0.05 and 0.3. slow
		->	'flip': 1, fast
"""
def batch_gen3(file_list, batch_size, aug = {}, shuffle_list = True, epochs = 1):
	cnt = 0
	batch = 0
	for filename in file_list:
		file = h5py.File(filename, 'r')
		input_shape = file['data'].shape
		output_shape = file['label'].shape


	im = np.zeros((batch_size, input_shape[1],input_shape[2],input_shape[3],input_shape[4]))
	gt = np.zeros((batch_size, output_shape[1],output_shape[2],output_shape[3],output_shape[4]))

	for i in range(epochs):
		if shuffle_list:
			random.shuffle(file_list)
		
		for filename in file_list:
			file = h5py.File(filename, 'r')
			input_im = file['data']
			output = file['label']

			for pat in range(input_im.shape[0]):
				
				im[batch, :, :, :, :] = input_im[pat, :, :, :, :]
				gt[batch, :, :, :, :] = output[pat, :, :, :, :]

				if 'rotate' in aug.keys():
					im[batch, :, :, :, :], gt[batch, :, :, :, :] = add_rotation3(im[batch, :, :, :, :] * 1, gt[batch, :, :, :, :] * 1, aug['rotate'])

				if 'affine' in aug:
					im[batch, :, :, :, :], gt[batch, :, :, :, :] = add_affine_transform3(im[batch, :, :, :, :] * 1, gt[batch, :, :, :, :] * 1, aug['affine'])

				if 'shift' in aug:
					im[batch, :, :, :, :], gt[batch, :, :, :, :] = add_shift3(im[batch, :, :, :, :] * 1, gt[batch, :, :, :, :] * 1, aug['shift'])
				
				if 'flip' in aug:
					im[batch, :, :, :, :], gt[batch, :, :, :, :] = add_flip3(im[batch, :, :, :, :] * 1, gt[batch, :, :, :, :] * 1)

					#
				im[im < -1024] = -1024
				im[im > 400] = 400

				im = im - np.amin(im)
				im = im / np.amax(im)
				im = im.astype(np.float32)
				gt = gt.astype(np.float32)
				
				batch = batch + 1
				if batch == batch_size:
					batch = 0
					yield im, gt
			file.close()

def batch_length(file_list):
	length = 0
	for filename in file_list:
		file = h5py.File(filename, 'r')
		input_im = file['data']
		for pat in range(input_im.shape[0]):
			length = length + 1
	print('images in generator:',length)
	return length	


