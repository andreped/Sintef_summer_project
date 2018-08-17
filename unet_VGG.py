import cv2
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from tqdm import tqdm
from scipy.ndimage import interpolation, zoom
from skimage.measure import regionprops, label
from stl import mesh
from mpl_toolkits import mplot3d
from skimage.measure import marching_cubes_lewiner
import h5py
import pydicom
'''
-INPUT
image_path: 	path to a folder with DICOM images

-RETURNS	
input_im:		3D array of ct images, in the range [-1024, 400]
res:			resolution in [z, xy]
offset:			offset list, for the raw/mhd files
'''
def import_images(image_path):
	file_list = []
	for file in os.listdir(image_path):
		if file.endswith('.dcm'):
			file_list.append(image_path + '/' + file)

	orig = []
	size = sitk.GetArrayFromImage(sitk.ReadImage(file_list[0])).shape[1:3]

	for image in file_list:
		itkimage = sitk.ReadImage(image)
		origin = np.array(list(reversed(itkimage.GetOrigin())))
		orig.append([origin[0],image])



	tmp = []
	for i in range(len(orig)):
		tmp.append(orig[i][0])
	z_res = (max(tmp) - min(tmp))/(len(tmp)-1)

	#offset
	ds = pydicom.dcmread(file_list[0])
	offset = [round(float(ds.ImagePositionPatient[0]), 1), round(float(ds.ImagePositionPatient[1]), 1), round(min(tmp), 1)]



	input_im = np.zeros((len(file_list), size[0], size[1]))
	cnt = 0
	for orig, path in sorted(orig):
		itkimage = sitk.ReadImage(path)
		ct_scan = sitk.GetArrayFromImage(itkimage)

		input_im[cnt, :, :] = ct_scan[0,:,:]
		spacing = np.array(list(reversed(itkimage.GetSpacing())))

		cnt = cnt + 1
	
	input_im[input_im < -1024] = -1024
	input_im[input_im > 400] = 400

	xy_res = spacing[1]

	#resolution 
	res = [z_res, xy_res]


	return input_im, res, offset

'''
-INPUT
ct_images:				a 3D array of ct images in the range [-1024, 400], and on the format Z-Y-X
unet_model_path:		path to the model
predict_with_overlap:	[bool , number of slices in overlap]. (TODO needs some love to make it work correctly)

-RETURNS
prediction:				3d array of the prediction	


'''
def pred_unet(ct_images, unet_model_path, predict_with_overlap = [False, 32]):
	model = load_model(unet_model_path, compile=False)
	model_dim = len(model.get_config()['layers'][0]['config']['batch_input_shape'])
	model_config = model.get_config()['layers'][0]['config']['batch_input_shape']
	#print('model:', unet_model_path.split('/')[-1])

	if model_dim != 5:
		print('this function only works for 3D Unet models')
		return -1

	# resize to the model config
	tmp = np.zeros((ct_images.shape[0], model_config[2], model_config[3], 1))
	for i in range(ct_images.shape[0]):
		tmp[i, :, :, 0] = cv2.resize(ct_images[i, :, :], (model_config[2], model_config[3]))
	ct_images_resized = tmp.copy()

	# predict with overlap
	if predict_with_overlap[0]:
		chunks = int(np.ceil(((ct_images_resized.shape[0] - model_config[1] ) / predict_with_overlap[1]) + 1))

		im = np.zeros((chunks, model_config[1], model_config[2], model_config[3], model_config[4]))
		pred_output_5d = np.zeros((chunks, model_config[1], model_config[2], model_config[3], 2))
		pred_output = np.zeros((ct_images_resized.shape[0], model_config[2], model_config[3], 2))
		
		for i in tqdm(range(chunks)):
			for j in range(model_config[1]):
				if model_config[1]*i +j >= ct_images_resized.shape[0]:
					continue
				im[i, j, :, :, 0] = ct_images_resized[(model_config[1]*i - predict_with_overlap[1]*i) +j, :, :,0]

			pred_output_5d[i] = model.predict(np.expand_dims(im[i, :, :, :, :], axis = 0))
		

		# add together the overlaps
		first_chunk = int(predict_with_overlap[1] + (model_config[1]/4)) # 48
		next_chunks = predict_with_overlap[1] # 32
		last_chunk = first_chunk

		for j in range(first_chunk):
			pred_output[j, :, :, :] = pred_output_5d[0, j, :, : ,:]

		for i in range(chunks-2):
			for j in range(next_chunks):
				pred_output[(j+first_chunk) + (i)*next_chunks, :, :, :] = pred_output_5d[i+1, j, :, : ,:]

		for j in range(last_chunk):
			if int(first_chunk + ((chunks-2)*next_chunks) + j) >= pred_output.shape[0]:
				break
			pred_output[first_chunk + ((chunks-2)*next_chunks) + j, :, :, :] = pred_output_5d[chunks-1, j, :, : ,:]

		prediction = np.zeros(ct_images.shape)
		for i in range(pred_output.shape[0]):
			prediction[i, :, :] = cv2.resize(pred_output[i, :, :, 1], (ct_images.shape[1], ct_images.shape[2]))

	#predict without overlap
	else:
		# number of chunks to create
		chunks = int(np.ceil(ct_images.shape[0] / model_config[1])) 
		
		# initilaize arrays
		pred_output_5d = np.zeros((chunks, model_config[1], model_config[2], model_config[3], 2))
		pred_output = np.zeros((model_config[1]*chunks, model_config[2], model_config[3]))
		im = np.zeros((chunks, model_config[1], model_config[2], model_config[3], model_config[4]))

		# place images in the 5d shape the unet takes in
		for i in range(chunks):
			for j in range(model_config[1]):
				if model_config[1]*i +j >= ct_images_resized.shape[0]:
					continue
				im[i, j, :, :, :] = ct_images_resized[model_config[1]*i +j, :, :, :]
		
		# predict
		for i in tqdm(range(chunks)):
			pred_output_5d[i, :, :, :, :] = model.predict(np.expand_dims(im[i, :, :, :, :], axis = 0))
		
		# reshape to one large stack
		for i in range(chunks):
			for j in range(pred_output_5d.shape[1]):
				pred_output[i*pred_output_5d.shape[1] +j, :, :] = pred_output_5d[i, j, :, : ,1]

		# resize to original size 
		prediction = np.zeros(ct_images.shape)
		for i in range(pred_output.shape[0]):
			if i >= ct_images.shape[0]:
				break
			prediction[i, :, :] = cv2.resize(pred_output[i, :, :], (ct_images.shape[1], ct_images.shape[2]))

	return prediction



'''
-INPUT
ct_images:		3d array of the ct images, [Z,Y,X]
prediction:		3d array of the prediction on the ct images
resolution:		list of the length in [mm] of the voxels, [z-len, xy-len] 
VGG_model_path:	path to the VGG model
thr:			the chosen threshold to binarise the prediction.

-RETURNS
prediction:		predicted malignancy, range 1 to 5
'''
def pred_VGG(ct_images, prediction, resolution, VGG_model_path, thr = 0.5):

	prediction[prediction <= thr] = 0
	prediction[prediction > thr] = 1

	model = load_model(VGG_model_path, compile=False)
	model_dim = len(model.get_config()['layers'][0]['config']['batch_input_shape'])
	model_config = model.get_config()['layers'][0]['config']['batch_input_shape']
	#print('model:', VGG_model_path.split('/')[-1])

	if model_dim != 5:
		print('this function only works for 3D VGG models')
		return -1

	scaling_factor = resolution[0] / resolution[1]
	size = 64 # shape of zyx
	mask = np.arange(1, 5.5, 0.5)
	
	labels = label(prediction)
	for reg in regionprops(labels, intensity_image = ct_images):

		input_st = np.zeros((1, size, size, size, 2), dtype=np.float32) * (-1024)
		center = np.round(reg.centroid).astype(int)
		#ct stack
		# takes out +- 64/2/scaling_factor from the center (z-dir), and 64/2 in xy-dir 
		tmp = ct_images[int(center[0]-((size/2)/scaling_factor)):int(center[0]+((size/2)/scaling_factor)), center[1]-int(size/2):center[1]+int(size/2), center[2]-int(size/2):center[2]+int(size/2)]
		tmp = zoom(tmp.copy(), zoom = [scaling_factor,1,1], order = 1)
	
		input_st[0, :tmp.shape[0], :tmp.shape[1], :tmp.shape[2], 0] = tmp

		# scale to [0,1]
		input_st[:, :, :, :, 0] = input_st[:, :, :, :, 0] - np.amin(input_st[:, :, :, :, 0])
		input_st[:, :, :, :, 0] = input_st[:, :, :, :, 0] / np.amax(input_st[:, :, :, :, 0])


		# remove all tumors from the prediction, exept the one in this region
		tot_tmp = prediction.copy()
		tot_tmp[labels != reg.label] = 0
		
		#binary stack
		tmp = tot_tmp[int(center[0]-((size/2)/scaling_factor)):int(center[0]+((size/2)/scaling_factor)), center[1]-int(size/2):center[1]+int(size/2), center[2]-int(size/2):center[2]+int(size/2)]

		tmp = zoom(tmp.copy(), zoom = [scaling_factor,1,1], order = 0)
		input_st[0, :tmp.shape[0], :tmp.shape[1], :tmp.shape[2], 1] = tmp


		pred = model.predict(input_st)
		p_max = np.amax(pred)
		pred[pred < p_max] = 0
		pred[pred != 0] = 1
		pred = pred*mask
		pred = np.unique(pred)[1]

		prediction[labels == reg.label] = pred

	return prediction



'''
-INPUT
prediction:	array of predictions
res:		resolution [z, xy]
save_path:	dir to save in
name:		name to save the stl file as	

'''

def pred_to_stl(prediction, res, save_path, name, thr = 0.3):

	#prediction[prediction < thr] = 0
	#prediction[prediction >= thr] = 1
	prediction = np.moveaxis(prediction.copy(),0,2)
	prediction = np.moveaxis(prediction.copy(),0,1)


	prediction = zoom(prediction.copy(), [res[1],res[1],res[0]], order = 1)


	verts, face, norm, val = marching_cubes_lewiner(prediction.astype(bool))

	surface = mesh.Mesh(np.zeros(face.shape[0], dtype=mesh.Mesh.dtype))
	for i, f in enumerate(face):
		for j in range(3):
			surface.vectors[i][j] = verts[f[j],:]

	surface.save(save_path + '/' + name + '.stl')


'''
- INPUT
prediction:	array of predictions
res:		resolution [z, xy]
save_path:	dir to save in
name:		name to save the raw and mhd file as	
offset: 	offset foor the mhd file

'''
def pred_to_raw(prediction, save_path, name, res, offset):
	#from ZYX to XYZ
	#prediction = np.moveaxis(prediction.copy(),0,2)
	#prediction = np.moveaxis(prediction.copy(),0,1)

	#to uint8
	prediction = (prediction.copy() * 255).astype(np.uint8)
	
	#save raw
	prediction.tofile(save_path +'/'+ name + '.raw')

	#tmp mhd file
	x = ['ObjectType = Image',
		'NDims = 3',
		'BinaryData = True',
		'BinaryDataByteOrderMSB = False',
		'CompressedData = False',
		'CenterOfRotation = 0 0 0',
		'ElementSpacing = 0.703125 0.703125 2.4812',
		'DimSize = 512 512 133',
		'ElementType = MET_UCHAR',
		'Modality = CT',
		'Offset =  -166 -171.7 -340',
		'ElementDataFile = test_raw.raw']

	#rewrite x
	for i in range(len(x)):
		#spacing
		if x[i].startswith('ElementSpacing'):
			spacing = x[i].split(' ')
			spacing[2:] = [str(res[1]), str(res[1]), str(res[0])]
			spacing = ' '.join(spacing)
			x[i] = spacing
		#dim size
		if x[i].startswith('DimSize'):
			dim = x[i].split(' ')
			dim[2:] = [str(prediction.shape[2]), str(prediction.shape[1]), str(prediction.shape[0])]
			dim = ' '.join(dim)
			x[i] = dim
		# offset
		if x[i].startswith('Offset'):
			off = x[i].split(' ')
			off[2:] = [str(offset[0]), str(offset[1]), str(offset[2])]
			off = ' '.join(off)
			x[i] = off
		# raw file
		if x[i].startswith('ElementDataFile'):
			elem = x[i].split(' ')
			elem[-1] = name+'.raw'
			elem = ' '.join(elem)
			x[i] = elem
	

	# save the mhd file
	f = open(save_path +'/'+ name + '.mhd', 'w')
	for i in range(len(x)):
		f.write(x[i]+'\n')
	f.close()










