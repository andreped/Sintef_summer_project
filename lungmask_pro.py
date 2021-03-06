

import numpy as np
from skimage.morphology import disk, binary_erosion, binary_closing, remove_small_holes, remove_small_objects
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops, label
from sklearn.cluster import KMeans
from skimage import morphology, measure
from scipy.ndimage.filters import median_filter
from skimage.filters import rank, threshold_otsu
from pytictoc import TicToc
from scipy.ndimage.interpolation import zoom
import os, sys
from cv2 import Canny, bilateralFilter, medianBlur, resize, dilate, erode, bitwise_not


'''
A set of functions relevant for lungmask. Both 2D and 3D. Also includes a scaling function to set dynamic (intensity) range 
to standard [0, 255] -> 8bit.
NB: Handle with caution. Was just made for testing and to achieve a simple and fast lungmask for 3D-data, where
accuracy in segmentation wasn't the main goal.
------------------------------------------------------------------------------------------------------------
Made by: André Pedersen
e-mail : ape107@post.uit.no
'''


# function which scales the image to standard 8-bit intensity range
# -> assumes numpy input, and returns input if single-values array input
def maxminscale(tmp):
	if (len(np.unique(tmp)) > 1):
		tmp = tmp - np.amin(tmp)
		tmp = 255/np.amax(tmp)*tmp
	return tmp


# Function that segments the lung in a 2D-image. Returns binary 2D-mask
# --- INPUT ---
# img : 2D numpy array
# morph : bool, whether or not to apply morphological step to include juxta-pleural nodules
# --- OUTPUT ---
# lungmask : binary numpy array where 1 correspond to lung, 0 else
## NB: Will also mask bladder for instance, so if you want to use this in 3D, you should use lungmask3D instead
def lungmask_pro(img, morph = True):

	# to avoid many tears
	img = img.copy()

	# set intensity range to standard (corners have problems) -> doesn't really matter, robust method, but looks better
	img[img <= -1024] = -1024
	img[img >= 1024] = 1024 # also this, because of circular region artifacts (actually also for the one above)

	# scale image to be in the range [0,255] -> because method robust enough to handle it -> smarter thresholds
	img = np.uint8(maxminscale(img))

	# keep scaled original image for masking later
	img_orig = img.copy()

	# blur img to easier being able to segment the lung using kmeans
	img = medianBlur(img, 5)

	# get dimensions of image
	row_size, col_size = img.shape

	# specify window for k-means to work on, such that you get the lung, and not the rest
	middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 

	# apply otsu's method to threshold image
	th = threshold_otsu(middle)
	thresh_img = np.where(img<th,1.0,0.0)

	# label each object from the filtering above and only get the lung including juxta-vascular nodules
	labels = label(thresh_img) # Different labels are displayed in different colors
	regions = regionprops(labels)
	good_labels = []
	for prop in regions:
	    B = prop.bbox
	    if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/10 and B[2]<col_size/10*9: # better window!
	        good_labels.append(prop.label)
	mask = np.ndarray([row_size,col_size],dtype=np.int8)
	mask[:] = 0

	# gets objects of interest from criteria defined and used above
	for N in good_labels:
	    mask = mask + np.where(labels==N,1,0)

	## fill regions surrounded by lung regions -> final ROI
	# to fill holes -> without filling unwanted larger region in the middle
	res = binary_fill_holes(mask)

	# need to filter out the main airways easily seperated from lung, because else they will affect the resulting algorithm
	res2 = remove_small_objects(label(res).astype(bool), min_size=800)
	res2[res2 > 0] = 1

	# separate each object in image (i.e. each lung part), and do morphology to include juxta-pleural nodules
	lungmask = np.zeros(res2.shape)
	labels = label(res2)
	for i in range(1,len(np.unique(labels))):
		tmp = np.zeros(labels.shape)
		tmp[labels == i] = 1

		# whether or not to apply morphology to fix lung boundary -> to include juxta-pleural nodules (nodules attached to lung boundary)
		if (morph == True):
			mask = dilate(np.uint8(tmp), disk(17)) # 17 : radius of 2D-disk
			mask = remove_small_objects(label(bitwise_not(maxminscale(mask))).astype(bool), min_size = 500).astype(int)
			mask[mask != 0] = -1
			mask = np.add(mask, np.ones(mask.shape))
			filled_tmp = erode(np.uint8(mask), disk(19))
		else:
			mask = remove_small_objects(label(bitwise_not(maxminscale(tmp))).astype(bool), min_size = 500).astype(int)
			mask[mask != 0] = -1
			filled_tmp = np.add(mask, np.ones(mask.shape))

		lungmask += filled_tmp

	return lungmask



# Function that masks the lung lung on a 3D-image stack. Returns binary 3D mask
# --- INPUT ---
# data : 3D-numpy array with dimensions (slice, (img x,y))
# morph : bool, whether or not to apply morphological step to include juxta-pleural nodules
# --- OUTPUT ---
# res  : binary 3D-numpy array where 1 correspond to lung, 0 else
## NB: Useless without 2D-lungmask, since uses it to segment
def lungmask3D(data, morph = True):

	# to avoid many tears
	tmp = data.copy()

	# apply lungmask on each slice
	for i in range(data.shape[0]):
		tmp[i] = lungmask_pro(data[i], morph)

	# remove smaller objects like bladder etc (if 3D object is smaller than 0.5 % of total volume)
	res = remove_small_objects(label(tmp).astype(bool), min_size = int(round(0.005*(data.shape[0]*data.shape[1]*data.shape[2])))).astype(int)
	res[res > 0] = 1

	return res















