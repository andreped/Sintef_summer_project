from unet_VGG import import_images, pred_unet, pred_VGG, pred_to_stl, pred_to_raw
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt
from lungmask_pro import lungmask3D
import numpy as np
import h5py
import cv2
import os
from scipy.ndimage import zoom

#path to folder with dicom
image_path = '/Users/medtek/Desktop/scripts and patient data/dataFromLIDC/LIDC-IDRI/LIDC-IDRI-0012/01-01-2000-50667/3000561-57842' # pat12
#path to unet model
unet_model_path = '/Users/medtek/Desktop/scripts and patient data/trained_nets/Unet_model.hd5'
#path to VGG classification model
VGG_model_path = '/Users/medtek/Desktop/scripts and patient data/trained_class_nets/model_classification_3d_06_08_2.hd5' 
#directory to save stl, raw and mhd files in
save_path = '/Users/medtek/Desktop/programs/stl_and_raw_files'
#name of the created files
name = 'pat_12'

if not os.path.exists(save_path):
	os.mkdir(save_path)

# read dicom
images, res, offset  = import_images(image_path)
#predict/segment with unet
unet_pred = pred_unet(images, unet_model_path)
#classifiy with VGG
VGG_pred = pred_VGG(images, unet_pred, res, VGG_model_path)


#---- To custusX ----
#save segmentation in stl format
pred_to_stl(unet_pred, res, save_path, name)
#save raw and mhd
pred_to_raw(unet_pred, save_path, name, res, offset)






#save lungmask as raw with mhd
mask = lungmask3D(images, morph = False)

images[images < -1024] = -1024
images[images >= 400] = 400
lung = images.copy()
lung[mask == 0] = np.amin(lung)
lung = lung - np.amin(lung)
lung = lung / np.amax(lung)

pred_to_raw(lung, save_path, name+'_lung', res, offset)


