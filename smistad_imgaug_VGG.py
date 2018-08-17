from __future__ import absolute_import
from __future__ import print_function

from skimage.transform import resize

try:
    from keras.preprocessing.image import Iterator
except:
    from tensorflow.python.keras.preprocessing.image import Iterator
import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
import scipy
import h5py
from scipy.ndimage.interpolation import map_coordinates, rotate
from scipy.ndimage.filters import gaussian_filter
import time




class UltrasoundImageIterator(Iterator):
    def __init__(self, generator, image_list, input_shape, output_shape, batch_size, shuffle, all_files_in_batch):
        self.image_list = image_list
        self.generator = generator
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.all_files_in_batch = all_files_in_batch
        epoch_size = len(image_list)
        if all_files_in_batch:
            epoch_size = len(image_list)*10
        super(UltrasoundImageIterator, self).__init__(epoch_size, batch_size, shuffle, None)

    def _get_sample(self, index):
        filename, file_index = self.image_list[index]
        file = h5py.File(filename, 'r')
        input = file['data'][file_index, :]

        tmp = np.expand_dims(file['label'][:],axis = 0)
        output = tmp
        #output = file['label'][file_index, :] # *****-fixed for classification of tumors with 1-d input
        file.close()
        return input, output

    def _get_random_sample_in_file(self, file_index):
        filename = self.image_list[file_index]
        file = h5py.File(filename, 'r')
        x = file['data']
        sample = np.random.randint(0, x.shape[0])
        #print('Sampling image', sample, 'from file', filename)
        input = file['data'][sample, :]

        tmp = np.expand_dims(file['label'][:],axis = 0)
        output = tmp
        #output = file['label'][sample, :] # *****-fixed for classification of tumors with 1-d input
        file.close()
        return input, output

    def next(self):

        with self.lock:
            index_array = next(self.index_generator)

        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array):

        #print('Making batch..')
        batch_x = np.zeros(tuple([len(index_array)] + list(self.input_shape))) #orig
        #batch_x = np.zeros(tuple([len(index_array)] + list((64,64,64,2))))
        batch_y = np.zeros(tuple([len(index_array)] + list(self.output_shape)))
        for batch_index, sample_index in enumerate(index_array):
            # Have to copy here in order to not modify original data
            if self.all_files_in_batch:
                input_image, output = self._get_random_sample_in_file(batch_index)
            else:
                input_image, output = self._get_sample(sample_index)
            new_input_image, output = self.generator.transform(input_image, output)
            
            # #scale to [0,1] and pad from 60 to 64 (gettostyle)
            new_input_image[:, :, :, 0][new_input_image[:, :, :, 0] < -1024] = -1024
            new_input_image[:, :, :, 0][new_input_image[:, :, :, 0] > 400] = 400
            new_input_image[:, :, :, 0] = new_input_image[:, :, :, 0] - np.amin(new_input_image[:, :, :, 0])
            new_input_image[:, :, :, 0] = new_input_image[:, :, :, 0] / (np.amax(new_input_image[:, :, :, 0]) + 10e-6)
            input_im = np.zeros((64,64,64,2)) * (-1024)
            input_im[:new_input_image.shape[0], :new_input_image.shape[1], :new_input_image.shape[2], :new_input_image.shape[3]] = new_input_image
            # #print(new_input_image[0,0,0,0])
            

            batch_x[batch_index] = input_im
            #batch_x[batch_index] = new_input_image # orig
            batch_y[batch_index] = output

        return batch_x, batch_y


CLASSIFICATION = 'classification'
SEGMENTATION = 'segmentation'


class UltrasoundImageGenerator():
    def __init__(self, filelist, all_files_in_batch=False):
        self.methods = []
        self.args = []
        self.crop_width_to = None
        self.image_list = []
        self.input_shape = None
        self.output_shape = None
        self.all_files_in_batch = all_files_in_batch

        if all_files_in_batch:
            file = h5py.File(filelist[0], 'r')
            self.input_shape = file['data'].shape[1:]
            self.output_shape = file['label'].shape[1:] 
            file.close()
            self.image_list = filelist
            return

        # Go through filelist
        for filename in filelist:
            # Open file to see how many images it has
            file = h5py.File(filename, 'r')
            if self.input_shape is None:
                self.input_shape = file['data'].shape[1:]
                self.output_shape = file['label'].shape[:]
                #self.output_shape = file['label'].shape[1:] # ***** -fixed for classification of tumors with 1-d input

                if len(self.output_shape) == 1:
                    self.problem_type = CLASSIFICATION
                else:
                    self.problem_type = SEGMENTATION

            samples = file['data'].shape[0]
            file.close()
            # Append a tuple to image_list for each image consisting of filename and index
            for i in range(samples):
                self.image_list.append((filename, i))
        print('Image generator with', len(self.image_list), ' image samples created')

    def add_intensity_scaling(self, min_scale, max_scale):
        self.methods.append(self._scale_intensity)
        self.args.append([min_scale, max_scale])

    def add_elastic_deformation(self, alpha=0.5, sigma=0.5):
        self.methods.append(self._elastic_deformation)
        self.args.append([alpha, sigma])

    def add_gaussian_shadow(self, sigma_x_min=0.1, sigma_x_max=0.5, sigma_y_min=0.1, sigma_y_max=0.9, strength_min=0.5, strength_max=0.8):
        self.methods.append(self._gaussian_shadow)
        self.args.append([sigma_x_min, sigma_x_max, sigma_y_min, sigma_y_max, strength_min, strength_max])

    def add_gamma_transformation(self, min=0.25, max=1.7):
        self.methods.append(self._gamma_transform)
        self.args.append([min, max])

    def add_cropping(self, target_width):
        self.methods.append(self._crop)
        self.args.append([target_width])
        self.crop_width_to = target_width

    def add_blurring(self, sigma_max=1.):
        self.methods.append(self._blur)
        self.args.append([sigma_max])

    def add_scaling(self, min_scaling=0.75, max_scaling=1.25):
        self.methods.append(self._scale)
        self.args.append([min_scaling, max_scaling])

    def add_label_flipping(self, labels, probability=0.1, flip='input'):
        """

        :param labels: a list of which labels are in the image
        :param probability:
        :param flip: String, either 'input' or 'output'
        :return:
        """
        self.methods.append(self._label_flipping)
        self.args.append([labels, probability, flip])

    def add_shifting(self, max_offset_x, max_offset_y, preserve_segmentation=False):
        """
        Shift image with random offset within the given bounds
        [-max_offset_x:max_offset_x, -max_offset_y:max_offset_y] and pads with 0
        :param max_offset_x:
        :param max_offset_y:
        :return:
        """
        self.methods.append(self._shift)
        self.args.append([max_offset_x, max_offset_y, preserve_segmentation])

    def add_rotation(self, max_angle):
        """
        Rotates image with arbitrarty angle in bounds [-max_angle:max_angle]
        :param max_angle: in degrees
        :return:
        """
        self.methods.append(self._rotate)
        self.args.append([max_angle])

    def add_flip(self):
        self.methods.append(self._flip)
        self.args.append(None)

    #### -------- 3D aug stuff-----
    def add_flip_3d(self):
        """
        Flips random axis
        works only for clasification problems!
        if problem_type = segmentation -> retuns the same arrays
        """
        self.methods.append(self._flip_3d)
        self.args.append(None)

    def add_rotate_3d(self, min_angle_3d, max_angle_3d):
        self.methods.append(self._rotate_3d)
        self.args.append([min_angle_3d, max_angle_3d])

    def _rotate_3d(self, input, output, min_angle_3d, max_angle_3d):
        angle_xy = np.random.uniform(min_angle_3d, max_angle_3d)
        angle_xz = np.random.uniform(min_angle_3d, max_angle_3d)

        input = rotate(input.copy(), angle_xy, axes = (1,2), reshape = False, mode = 'nearest', order = 1)
        input = rotate(input.copy(), angle_xz, axes = (1,2), reshape = False, mode = 'nearest', order = 1)
        input[:, :, :, 1][input[:, :, :, 1] <= 0.5] = 0
        input[:, :, :, 1][input[:, :, :, 1] > 0.5] = 1
        

        return input, output

    def _flip_3d(self, input, output):
        flip_ax = np.random.random_integers(0, high = 3)
        if self.problem_type == SEGMENTATION:
            return input, output
        if flip_ax == 3:
            return input, output
        else:
            return np.flip(input, flip_ax), output

    #### ----------------------------

    def _scale(self, input, output, min_scaling, max_scaling):
        scaling_factor = np.random.uniform(min_scaling, max_scaling)

        def crop_or_fill(image, shape):
            image = np.copy(image)
            for dimension in range(2):
                if image.shape[dimension] > shape[dimension]:
                    # Crop
                    if dimension == 0:
                        image = image[:shape[0], :]
                    elif dimension == 1:
                        image = image[:, :shape[1], :]
                else:
                    # Fill
                    if dimension == 0:
                        new_image = np.zeros((shape[0], image.shape[1], shape[2]))
                        new_image[:image.shape[0], :, :] = image
                    elif dimension == 1:
                        new_image = np.zeros((shape[0], shape[1], shape[2]))
                        new_image[:, :image.shape[1], :] = image
                    image = new_image
            return image

        input = crop_or_fill(scipy.ndimage.zoom(input, scaling_factor, order=1), input.shape)
        if self.problem_type == SEGMENTATION:
            output = crop_or_fill(scipy.ndimage.zoom(output, scaling_factor, order=0), output.shape)

        return input, output

    def _rotate(self, input, output, max_angle):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html
        angle = np.random.randint(-max_angle, max_angle)

        input = scipy.ndimage.rotate(input, angle, order=1, reshape=False) # Using order=2 here gives incorrect results

        if self.problem_type == SEGMENTATION:
            # Transform output
            output[:, :, 0] = np.ones(output.shape[:2]) # Clear background
            for label in range(1, output.shape[2]):
                segmentation = scipy.ndimage.rotate(output[:, :, label], angle, order=0, reshape=False).reshape(output.shape[:2])
                output[:, :, label] = segmentation

            # Remove segmentation from other labels
            for label in range(1, output.shape[2]):
                for label2 in range(output.shape[2]):
                    if label2 == label:
                        continue
                    output[output[:, :, label] == 1, label2] = 0

        return input, output

    def _flip(self, input, output):
        if np.random.choice([True, False]):
            if self.problem_type == SEGMENTATION:
                return np.fliplr(input), np.fliplr(output)
            else:
                return np.fliplr(input), output
        else:
            return input, output

    def _gamma_transform(self, input, output, min, max):
        gamma = np.random.uniform(min, max)
        return np.clip(np.power(input, gamma), 0, 1), output

    def _label_flipping(self, input, output, labels, probability, flip='input'):
        # For each pixel flip label of output with a probability
        if flip == 'input':
            size = input.shape[:2]
        else:
            size = output.shape[:2]

        probabilities = np.random.rand(size[0], size[1])
        new_labels = np.random.randint(0, high=len(labels), size=size)

        new_input = input
        new_output = output
        x, y = np.nonzero(probabilities < probability)
        for i in range(len(x)):
            if flip == 'input':
                new_input[x[i], y[i], :] = 0
                new_input[x[i], y[i], new_labels[x[i], y[i]]] = 1
            else:
                new_output[x[i], y[i], :] = 0
                new_output[x[i], y[i], new_labels[x[i], y[i]]] = 1

        return new_input, new_output

    def _shift(self, input, output, offset_x, offset_y, preserve_segmentation):
        start = time.time()
        non = lambda s: s if s < 0 else None
        mom = lambda s: max(0, s)

        min_offset_y = -offset_y
        max_offset_y = offset_y
        min_offset_x = -offset_x
        max_offset_x = offset_x
        if self.problem_type == SEGMENTATION and preserve_segmentation:
            # TODO this can be done smarter to find the minimal and maximal offsets possible
            # Check if any segmentation at bottom
            if np.sum(output[0:abs(min_offset_y), :, 1:]) > 0:
                min_offset_y = 0
            # Check if any segmentation at top
            if np.sum(output[-max_offset_y:-1, :, 1:]) > 0:
                max_offset_y = 0
            # Check if any segmentation at left
            if np.sum(output[:, 0:abs(min_offset_x), 1:]) > 0:
                min_offset_x = 0
            # Check if any segmentation at right
            if np.sum(output[:, -min_offset_x:-1, 1:]) > 0:
                max_offset_x = 0

        if min_offset_x == max_offset_x:
            ox = 0
        else:
            ox = np.random.randint(min_offset_x, max_offset_x)
        if min_offset_y == max_offset_y:
            oy = 0
        else:
            oy = np.random.randint(min_offset_y, max_offset_y)

        shift_input = np.zeros_like(input, dtype=np.float32)
        shift_input[mom(oy):non(oy), mom(ox):non(ox)] = input[mom(-oy):non(-oy), mom(-ox):non(-ox)]

        if self.problem_type == SEGMENTATION:
            shift_output = np.zeros_like(output, dtype=np.float32)
            label_count = output.shape[-1]
            shift_output[:, :, 0] = 1
            for label in range(1, label_count):
                shift_output[mom(oy):non(oy), mom(ox):non(ox), label] = output[mom(-oy):non(-oy), mom(-ox):non(-ox), label]
                shift_output[shift_output[:, :, label] == 1, 0] = 0
        else:
            shift_output = output

        end = time.time()
        #print('Shift time', (end-start))

        return shift_input, shift_output


    def _scale_intensity(self, input_image, output, min_scale, max_scale):
        # Scale image with a randon factor
        #print('before scaling:', input_image.dtype)
        input_image = input_image*np.random.uniform(min_scale, max_scale, size=1).astype(np.float32)
        # TODO should probably check here that image doesn't become too dark or bright

        # Clip to [0, 1]
        np.clip(input_image, 0, 1)
        #print('scaling:', input_image.dtype)

        return input_image, output

    def _modify_label(self, output, delta_x, image_width, crop_width):
        nr_of_objects = output[0].shape[0]
        for object_id in range(nr_of_objects):
            x_position = object_id*2
            y_position = object_id*2 + 1
            if output[0][object_id] == 1:  # Object is in image
                bounding_box_center_x = output[1][x_position]
                bounding_box_center_x *= image_width  # From normalized to pixel coordinate
                bounding_box_center_x = bounding_box_center_x - delta_x  # To cropped image coordinates

                # Check if artery is inside the cropped image
                if bounding_box_center_x < 0 or bounding_box_center_x >= crop_width:
                    output[0][object_id] = 0
                    output[1][x_position] = 0
                    output[1][y_position] = 0
                    output[2][x_position] = 0
                    output[2][y_position] = 0
                else:
                    # Need to handle cases were part of the vessel is outside the image
                    vessel_width = output[2][x_position] * image_width
                    if bounding_box_center_x + vessel_width * 0.5 > crop_width:
                        # Part of vessel is out of the image on the right side
                        pixels_over = bounding_box_center_x + vessel_width * 0.5 - crop_width
                        new_vessel_width = vessel_width - pixels_over
                        box_left_pos = bounding_box_center_x - vessel_width * 0.5
                        bounding_box_center_x = box_left_pos + new_vessel_width * 0.5
                        output[2][x_position] = new_vessel_width / crop_width
                        vessel_width = new_vessel_width
                    elif bounding_box_center_x - vessel_width * 0.5 < 0:
                        # Part of vessel is out of the image on the left side
                        pixels_below = abs(bounding_box_center_x - vessel_width * 0.5)
                        new_vessel_width = vessel_width - pixels_below
                        bounding_box_center_x = new_vessel_width * 0.5
                        vessel_width = new_vessel_width

                    # Update center_x position and width, and normalize it to new width
                    output[1][x_position] = bounding_box_center_x / crop_width
                    output[2][x_position] = vessel_width / crop_width

        return output

    def _crop(self, input_image, output, target_width):
        # TODO Crop image to target_width
        # Do random cropping
        image_width = input_image.shape[1]
        delta_x = np.random.randint(0, image_width - target_width)
        input_image = input_image[:, delta_x:delta_x + target_width]
        # Modify label for cropped image coordinate system
        output = self._modify_label(output, delta_x, image_width, target_width)

        return input_image, output

    def _elastic_deformation(self, image, output, alpha, sigma, random_state=None):
        import cv2
        """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis", in
           Proc. of the International Conference on Document Analysis and
           Recognition, 2003.
        """
        start = time.time()
        original_shape = image.shape
        original_output_shape = output.shape
        image = image[:, :, 0].astype(np.float32)
        #assert len(image.shape) == 2

        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape
        sigma = np.random.uniform(0.08, 0.1)*shape[1]#np.random.uniform(shape[0] * 0.5, shape[1] * 0.5)
        alpha = np.random.uniform(0.8, 1.0)*shape[1]
        #print('Parameters: ', alpha, sigma)

        blur_size = int(4 * sigma) | 1
        dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma)*alpha
        dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma)*alpha
        #dx = gaussian_filter(, sigma, mode="constant", cval=0) * alpha
        #dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dz = np.zeros_like(dx)

        #x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        #indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

        # Transform output
        output[:, :, 0] = np.ones(output.shape[:2]) # Clear background
        for label in range(1, output.shape[2]):
            #segmentation = cv2.remap(output[:, :, label], dx, dy, interpolation=cv2.INTER_LINEAR).reshape(output.shape[:2])
            segmentation = map_coordinates(output[:, :, label], indices, order=0).reshape(output.shape[:2])
            output[:, :, label] = segmentation

        # Remove segmentation from other labels
        for label in range(1, output.shape[2]):
            for label2 in range(output.shape[2]):
                if label2 == label:
                    continue
                output[output[:, :, label] == 1, label2] = 0

        #input = cv2.remap(image, dx, dy, interpolation=cv2.INTER_LINEAR).reshape(original_shape)
        input = map_coordinates(image, indices, order=1).reshape(original_shape)
        end = time.time()

        #print('Elastic deformation time', (end - start))

        return input, output

    def _blur(self, input_image, output, sigma_max):
        if np.random.choice([True, False]):
            # Random sigma
            sigma = np.random.uniform(0., sigma_max)
            input_image = scipy.ndimage.filters.gaussian_filter(input_image, sigma)
        return input_image, output

    def _gaussian_shadow(self, input_image, output, sigma_x_min, sigma_x_max, sigma_y_min, sigma_y_max, strength_min, strength_max):
        size = input_image.shape[0:2]
        for channel in [0]:#input_image.shape[3]:
            x, y = np.meshgrid(np.linspace(-1, 1, size[1], dtype=np.float32), np.linspace(-1, 1, size[0], dtype=np.float32), copy=False)
            x_mu = np.random.uniform(-1.0, 1.0, 1).astype(np.float32)
            y_mu = np.random.uniform(-1.0, 1.0, 1).astype(np.float32)
            sigma_x = np.random.uniform(sigma_x_min, sigma_x_max, 1).astype(np.float32)
            sigma_y = np.random.uniform(sigma_y_min, sigma_y_max, 1).astype(np.float32)
            strength = np.random.uniform(strength_min, strength_max, 1).astype(np.float32)
            g = 1.0 - strength * np.exp(-((x - x_mu) ** 2 / (2.0 * sigma_x ** 2) + (y - y_mu) ** 2 / (2.0 * sigma_y ** 2)), dtype=np.float32)

            input_image[:, :, channel] = input_image[:, :, channel]*np.reshape(g, size)

        return input_image, output

    def flow(self, batch_size, shuffle=True):

        return UltrasoundImageIterator(self, self.image_list, self.input_shape, self.output_shape, batch_size, shuffle, self.all_files_in_batch)

    def transform(self, input_image, output):
        input_image = input_image.astype(np.float32)
        output = output.astype(np.float32)
        # Loop over all methods that have been added and execute them (NOTE: assumes they can be run in any order)
        # TODO implement some sort of priority order
        for i, method in enumerate(self.methods):
            if self.args[i]:
                input_image, output = method(input_image, output, *self.args[i])
            else:
                input_image, output = method(input_image, output)

        return input_image, output

    def get_size(self):
        if self.all_files_in_batch:
            return 10*len(self.image_list)
        else:
            return len(self.image_list)