import cv2
import os,sys
import h5py
import SimpleITK as sitk
import numpy as np
from image_functions import object_filler, maxminscale
import matplotlib.pyplot as plt
import xml.etree.cElementTree as et
from image_functions import object_filler
from tensorflow.python.keras.models import load_model
from tqdm import tqdm

# --- A generator that takes in the path to the lidc dataset and the model path, and yields the ct images, 
# the ground truth, and the prediction from the  model. ----

# model_path:	path to the model that predicts
# start_path:	path to the lidc dataset
# pred_only_gt:	only predicts on images that have ground truths in them
# maxmin_input:	whether or not the model has trained on maxmin sacled images
# only:			include only the images in only
# exclude:		exclude the images in exclude



#yields: input_im, out, pred_output, pat  IF predict = True
#ELSE yields: input_im, out, pat


def predict_gen(model_path, 
			start_path, 
			pred_only_gt = False,
			maxmin_input = False,
			one_zero_scale = False,
			only = [], 
			exclude = [],
			predict = True):

	model = load_model(model_path, compile=False)
	model_dim = len(model.get_config()['layers'][0]['config']['batch_input_shape'])
	model_config = model.get_config()['layers'][0]['config']['batch_input_shape']
	print('model:', model_path.split('/')[-1],', model dimensions:',model_dim-2)


	s = '/'
	dcm_file_path = []
	dcm_files = {}
	xml_files = {}
	print()

	for dir1 in os.listdir(start_path):
		if dir1.startswith('.DS'):
			continue
		path1 = start_path +s+ dir1

		for dir2 in os.listdir(path1):
			if dir2.startswith('.DS'):
				continue
			pat_nr = int(dir2.split('-')[2])
			path2 = path1 +s+ dir2

			if len(only)> 20:
				sys.stdout.write("\033[F")
				print('reading through files, at patient_nr:', pat_nr)

			if int(pat_nr) in exclude: 		
				continue

			if only:
				if int(pat_nr) not in only:
					continue

			for dir3 in os.listdir(path2):
				if dir3.startswith('.DS'):
					continue
				path3 = path2 +s+ dir3	
				for dir4 in os.listdir(path3):
					if dir4.startswith('.DS'):
						continue
					path4 = path3 +s+ dir4

					for file in os.listdir(path4):
						if file.endswith('dcm') and len(os.listdir(path4)) > 10:
							dcm_file_path.append(path4 + s + file)
						if file.endswith('.xml') and len(os.listdir(path4)) > 10:
							xml_file_path = path4 + s + file

			dcm_files[pat_nr] = dcm_file_path
			xml_files[pat_nr] = xml_file_path
			dcm_file_path = []


	for pat in dcm_files.keys():
		orig = [] 

		for image in dcm_files[pat]:
			itkimage = sitk.ReadImage(image)
			origin = np.array(list(reversed(itkimage.GetOrigin())))
			orig.append([origin[0],image])

		#resolution
		z_res = np.abs(sorted(orig)[0][0] - sorted(orig)[1][0])

		tree = et.parse(xml_files[pat])
		root = tree.getroot()

		nodDicSmall = {}		# nodule <= 3mm dict
		nodDicLarge = {}		#large nodule dict
		nonNodDic = {}			#non-nodule dict
		nod_list = []
		non_nod_list = []
		nod_small_list = []

		# Read all unblinded read sessions
		readSes = root.findall('{http://www.nih.gov}readingSession') 
		for doctor in range(len(readSes)):

			#get real nodules(tumors)
			nodule = readSes[doctor].findall('{http://www.nih.gov}unblindedReadNodule')

			for i in range(len(nodule)):
				for roi in nodule[i].findall('{http://www.nih.gov}roi'):
					# single pixel nodules
					if len(roi) <= 5: 
						zValue = roi.find('{http://www.nih.gov}imageZposition')
						for edgeMap in roi.findall('{http://www.nih.gov}edgeMap'):	
							xValue = edgeMap.find('{http://www.nih.gov}xCoord')
							yValue = edgeMap.find('{http://www.nih.gov}yCoord')

							nodDicSmall.setdefault(float(zValue.text),[]).append([int(xValue.text),int(yValue.text)])
							
					else:
						zValue = roi.find('{http://www.nih.gov}imageZposition')
						for edgeMap in roi.findall('{http://www.nih.gov}edgeMap'):	
							xValue = edgeMap.find('{http://www.nih.gov}xCoord')
							yValue = edgeMap.find('{http://www.nih.gov}yCoord')

							nodDicLarge.setdefault(float(zValue.text),[]).append([int(xValue.text),int(yValue.text)])
			nod_list.append(nodDicLarge)
			nod_small_list.append(nodDicSmall)
			nodDicSmall = {}
			nodDicLarge = {}				
			
			#get non-nodules
			nonNodule = readSes[doctor].findall('{http://www.nih.gov}nonNodule')

			for i in range(len(nonNodule)):
				zValue = nonNodule[i].find('{http://www.nih.gov}imageZposition')
				for locus in nonNodule[i].findall('{http://www.nih.gov}locus'):
					xValue = locus.find('{http://www.nih.gov}xCoord')
					yValue = locus.find('{http://www.nih.gov}yCoord')
					nonNodDic.setdefault(float(zValue.text),[]).append([int(xValue.text),int(yValue.text)])
			non_nod_list.append(nonNodDic)
			nonNodDic = {}
	


		output_im = np.zeros((4, len(dcm_files[pat]), 512, 512,2),dtype = np.uint8)

		input_im = np.zeros((len(dcm_files[pat]), 512, 512,1))
		cnt = 0
		for orig, path in sorted(orig):
			itkimage = sitk.ReadImage(path)
			ct_scan = sitk.GetArrayFromImage(itkimage)
			im = ct_scan[0,:,:]
			input_im[cnt, :, :,0] = im
			#resolution
			spacing = np.array(list(reversed(itkimage.GetSpacing())))
			xy_res = spacing[1]



			# large nodules
			for list_ in range(len(nod_list)):
				if orig in nod_list[list_].keys():

					for j in range(len(nod_list[list_][orig])):
						output_im[list_, cnt, nod_list[list_][orig][j][1], nod_list[list_][orig][j][0], 1] = 255

					output_im[list_, cnt, :, :, 1] = object_filler(output_im[list_, cnt, :, :, 1], (0,0))
					#output_im[list_, cnt, :, :, 0] = cv2.medianBlur(np.uint8(output_im[list_, cnt, :, :,0]), 7) 

			cnt += 1	

		out = np.zeros((len(dcm_files[pat]), 512, 512,2))

		out[:,:,:,1] = np.add(np.add(output_im[0, :, :, :, 1].astype(int),output_im[1, :, :, :, 1].astype(int)),
					np.add(output_im[2, :, :, :, 1].astype(int), output_im[3, :, :, :, 1].astype(int))
					)
		out[:,:,:,1][out[:,:,:,1] < 300] = 0
		out[:,:,:,1][out[:,:,:,1] >= 300] = 1
		#one-hot	
		
		out[:,:,:,0][out[:,:,:,1] == 1] = 0
		out[:,:,:,0][out[:,:,:,1] == 0] = 1
		
		#resolution
		res = [z_res, xy_res]

		if predict:
			# ----- predicting 2d Unet ------
			if model_dim == 4:
				im = np.zeros((1,input_im.shape[1],input_im.shape[2],input_im.shape[3]))
				pred_output = np.zeros((out.shape[0],out.shape[1],out.shape[2],out.shape[3]))
				for i in tqdm(range(input_im.shape[0])):
					im[0, :, :, :] = input_im[i, :, :, :] 

					if pred_only_gt:
						if np.count_nonzero(out[i, :, :, 1]) == 0:
							continue
					pred_output[i, :, :, :] = model.predict(im)

			# ----- predicting 3d Unet -----
			if model_dim == 5:
				inp = np.zeros((input_im.shape[0],256,256,1))
				output = np.zeros((input_im.shape[0],256,256,2))
				for i in range(input_im.shape[0]):

					if maxmin_input:
						inp[i, :, :, 0] = cv2.resize(maxminscale(input_im[i, :, :, 0]), (256,256))
					else:
						inp[i, :, :, 0] = cv2.resize(input_im[i, :, :, 0], (256,256))

					output[i, :, :, 0] = cv2.resize(out[i, :, :, 0], (256,256),interpolation = cv2.INTER_NEAREST)
					output[i, :, :, 1] = cv2.resize(out[i, :, :, 1], (256,256), interpolation = cv2.INTER_NEAREST)

				input_im = inp.copy()
				out = output.copy()

				chunks = int(np.ceil(input_im.shape[0] / model_config[1]))

				pred_output = np.zeros((model_config[1]*chunks, model_config[2], model_config[3], 2))
				pred_output_5d = np.zeros((chunks, model_config[1], model_config[2], model_config[3], 2))
				im = np.zeros((chunks, model_config[1], model_config[2], model_config[3], model_config[4]))

				for i in range(chunks):
					for j in range(model_config[1]):
						if model_config[1]*i +j >= input_im.shape[0]:
							continue

						im[i, j, :, :, :] = input_im[model_config[1]*i +j, :, :, :]

				for i in tqdm(range(chunks)):

					if pred_only_gt:
						if np.count_nonzero(out[model_config[1]*i: (i+1)*model_config[1], :, :, 1]) == 0:
							continue

					pred_output_5d[i, :, :, :, :] = model.predict(np.expand_dims(im[i, :, :, :, :], axis = 0))
				
				for i in range(chunks):
					for j in range(pred_output_5d.shape[1]):
						pred_output[i*pred_output_5d.shape[1] +j, :, :, :] = pred_output_5d[i, j, :, : ,:]


	
			yield input_im, out, pred_output, pat, res
		else:
			yield input_im, out, pat, res


# function that returns the diffferent radiologists opinions
# start_path:	path to the folder that contains the lidc images
# only:			list of integers, only include these patients, 
# get_small:	bool, include small tumors
# get_non:		bool, include non nodules. 
def rad_opinions(start_path, only = [], exclude = [], get_small = True, get_non = True):
	s = '/'
	dcm_file_path = []
	dcm_files = {}
	xml_files = {}

	for dir1 in os.listdir(start_path):
		if dir1.startswith('.DS'):
			continue
		path1 = start_path +s+ dir1

		for dir2 in os.listdir(path1):
			if dir2.startswith('.DS'):
				continue
			pat_nr = int(dir2.split('-')[2])
			path2 = path1 +s+ dir2

			if int(pat_nr) in exclude: 		
				continue

			if only:
				if int(pat_nr) not in only:
					continue

			for dir3 in os.listdir(path2):
				if dir3.startswith('.DS'):
					continue
				path3 = path2 +s+ dir3	
				for dir4 in os.listdir(path3):
					if dir4.startswith('.DS'):
						continue
					path4 = path3 +s+ dir4

					for file in os.listdir(path4):
						if file.endswith('dcm') and len(os.listdir(path4)) > 10:
							dcm_file_path.append(path4 + s + file)
						if file.endswith('.xml') and len(os.listdir(path4)) > 10:
							xml_file_path = path4 + s + file

			dcm_files[pat_nr] = dcm_file_path
			xml_files[pat_nr] = xml_file_path
			dcm_file_path = []


	pat = list(dcm_files.keys())
	pat = pat[0]
	orig = [] 

	for image in dcm_files[pat]:
		itkimage = sitk.ReadImage(image)
		origin = np.array(list(reversed(itkimage.GetOrigin())))
		orig.append([origin[0],image])


	tree = et.parse(xml_files[pat])
	root = tree.getroot()

	nodDicSmall = {}		# nodule <= 3mm dict
	nodDicLarge = {}		#large nodule dict
	nonNodDic = {}			#non-nodule dict
	nod_list = []
	non_nod_list = []
	nod_small_list = []

	readSes = root.findall('{http://www.nih.gov}readingSession') 
	for doctor in range(len(readSes)):

		nodule = readSes[doctor].findall('{http://www.nih.gov}unblindedReadNodule')

		for i in range(len(nodule)):
			for roi in nodule[i].findall('{http://www.nih.gov}roi'):
				if len(roi) <= 5: 
					zValue = roi.find('{http://www.nih.gov}imageZposition')
					for edgeMap in roi.findall('{http://www.nih.gov}edgeMap'):	
						xValue = edgeMap.find('{http://www.nih.gov}xCoord')
						yValue = edgeMap.find('{http://www.nih.gov}yCoord')

						nodDicSmall.setdefault(float(zValue.text),[]).append([int(xValue.text),int(yValue.text)])
						
				else:
					zValue = roi.find('{http://www.nih.gov}imageZposition')
					for edgeMap in roi.findall('{http://www.nih.gov}edgeMap'):	
						xValue = edgeMap.find('{http://www.nih.gov}xCoord')
						yValue = edgeMap.find('{http://www.nih.gov}yCoord')

						nodDicLarge.setdefault(float(zValue.text),[]).append([int(xValue.text),int(yValue.text)])
		nod_list.append(nodDicLarge)
		nod_small_list.append(nodDicSmall)
		nodDicSmall = {}
		nodDicLarge = {}				
		
		nonNodule = readSes[doctor].findall('{http://www.nih.gov}nonNodule')

		for i in range(len(nonNodule)):
			zValue = nonNodule[i].find('{http://www.nih.gov}imageZposition')
			for locus in nonNodule[i].findall('{http://www.nih.gov}locus'):
				xValue = locus.find('{http://www.nih.gov}xCoord')
				yValue = locus.find('{http://www.nih.gov}yCoord')
				nonNodDic.setdefault(float(zValue.text),[]).append([int(xValue.text),int(yValue.text)])
		non_nod_list.append(nonNodDic)
		nonNodDic = {}



	output_im = np.zeros((4, len(dcm_files[pat]), 512, 512,1),dtype = np.uint8)

	input_im = np.zeros((len(dcm_files[pat]), 512, 512,1))
	cnt = 0
	for orig, path in sorted(orig):
		itkimage = sitk.ReadImage(path)
		ct_scan = sitk.GetArrayFromImage(itkimage)
		im = ct_scan[0,:,:]
		input_im[cnt, :, :,0] = im

		for list_ in range(len(nod_list)):
			if orig in nod_list[list_].keys():

				for j in range(len(nod_list[list_][orig])):
					output_im[list_, cnt, nod_list[list_][orig][j][1], nod_list[list_][orig][j][0], 0] = 255

		if get_small:
			for list_ in range(len(nod_small_list)):
				if orig in nod_small_list[list_].keys():

					for j in range(len(nod_small_list[list_][orig])):
						output_im[list_, cnt, :, :,0] = cv2.circle(output_im[list_, cnt, :, :, 0], (nod_small_list[list_][orig][j][0], nod_small_list[list_][orig][j][1]),2,255)
						output_im[list_, cnt, :, :,0] = cv2.putText(output_im[list_, cnt, :, :, 0],'small nodule', (nod_small_list[list_][orig][j][0], nod_small_list[list_][orig][j][1] + 20),4, 0.5,255)
		if get_non:	
			for list_ in range(len(non_nod_list)):
				if orig in non_nod_list[list_].keys():

					for j in range(len(non_nod_list[list_][orig])):
						output_im[list_, cnt, :, :,0] = cv2.circle(output_im[list_, cnt, :, :, 0], (non_nod_list[list_][orig][j][0], non_nod_list[list_][orig][j][1]),3,255)
						output_im[list_, cnt, :, :,0] = cv2.putText(output_im[list_, cnt, :, :, 0],'non nodule', (non_nod_list[list_][orig][j][0], non_nod_list[list_][orig][j][1] + 20),4,0.5,255)
		cnt += 1	


	return output_im