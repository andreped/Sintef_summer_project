#clasification - dataset optmizer test

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
from pytictoc import TicToc
from sklearn import metrics
from skimage.measure import regionprops, label, marching_cubes_lewiner, mesh_surface_area, find_contours
from scipy.ndimage.interpolation import zoom
from scipy.spatial.distance import euclidean
from from_Andre.lungmask_pro import lungmask_pro

'''
only: 		only include these patients, if empty -> all will be included
exclude: 	exclude there patients 

'''

def classification_dataset_gen(start_path, end_path, only = [], exclude = []):
	


	""""
	Extracting the path to the dicom files and the xml files. 
	Places them in dicts with patient number as key  
	"""
	dcm_file_path = []
	dcm_files = {}
	xml_files = {}
	s = '/'
	print()


	for dir1 in os.listdir(start_path):
		if not dir1.startswith('LIDC-IDRI'):
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


	mask = np.arange(1,5.5,0.5)

	"""
	For each patient 
	"""
	for pat in tqdm(dcm_files.keys()):

		#Get the origin for that patient  
		orig = [] 
		for image in dcm_files[pat]:
			itkimage = sitk.ReadImage(image)
			origin = np.array(list(reversed(itkimage.GetOrigin())))
			orig.append([origin[0],image])

		z_res = np.abs(sorted(orig)[0][0] - sorted(orig)[1][0])

		"""
		parse the xml file 

		"""
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
				for ID in nodule[i].findall('{http://www.nih.gov}noduleID'):
					n_id = ID.text

				malignancy = []
				for char in  nodule[i].findall('{http://www.nih.gov}characteristics'):
					malig = char.find('{http://www.nih.gov}malignancy')
					if malig != None:
						malignancy = malig.text
				
				for roi in nodule[i].findall('{http://www.nih.gov}roi'):
					# single pixel nodules
					if len(roi) <= 5: 
						zValue = roi.find('{http://www.nih.gov}imageZposition')
						for edgeMap in roi.findall('{http://www.nih.gov}edgeMap'):	
							xValue = edgeMap.find('{http://www.nih.gov}xCoord')
							yValue = edgeMap.find('{http://www.nih.gov}yCoord')

							nodDicSmall.setdefault(float(zValue.text),[]).append([int(xValue.text),int(yValue.text),False])
							
					else:
						zValue = roi.find('{http://www.nih.gov}imageZposition')
						for edgeMap in roi.findall('{http://www.nih.gov}edgeMap'):	
							xValue = edgeMap.find('{http://www.nih.gov}xCoord')
							yValue = edgeMap.find('{http://www.nih.gov}yCoord')

							if not malignancy:
								malignancy = 0
							nodDicLarge.setdefault(float(zValue.text),[]).append([int(xValue.text),int(yValue.text),malignancy])
	

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



		"""
		Read the dicom files, and create the ground truth 
		"""
		output_im = np.zeros((4, len(dcm_files[pat]), 512, 512),dtype = np.uint8)
		output_mal = np.zeros((output_im.shape))
		input_im = np.zeros((len(dcm_files[pat]), 512, 512))
		cnt = 0

		for orig, path in sorted(orig):
			itkimage = sitk.ReadImage(path)
			ct_scan = sitk.GetArrayFromImage(itkimage)
			im = ct_scan[0,:,:]
			input_im[cnt, :, :] = im
			spacing = np.array(list(reversed(itkimage.GetSpacing())))
			
			if spacing[1] == spacing[2]:
				xy_res = spacing[1]
			else:
				xy_res = (spacing[1] + spacing[2]) / 2


			# large nodules
			for list_ in range(len(nod_list)):
				if orig in nod_list[list_].keys():

					for j in range(len(nod_list[list_][orig])):
						output_im[list_, cnt, nod_list[list_][orig][j][1], nod_list[list_][orig][j][0]] = 255

					output_im[list_, cnt, :, :] = object_filler(output_im[list_, cnt, :, :], (0,0))
					output_mal[list_, cnt, :, :][output_im[list_, cnt, :, :]  == 255] = 1
					output_mal[list_, cnt, :, :] = output_mal[list_, cnt, :, :] * int(nod_list[list_][orig][0][2])


			cnt += 1	

		# 2 doctors agree on each pixel (50% consensus)
		tot = np.zeros((len(dcm_files[pat]), 512, 512))
		tot[:,:,:] = np.add(np.add(output_im[0, :, :, :].astype(int),output_im[1, :, :, :].astype(int)),
					np.add(output_im[2, :, :, :].astype(int), output_im[3, :, :, :].astype(int))
					)
		tot[:,:,:][tot[:,:,:] < 300] = 0
		tot[:,:,:][tot[:,:,:] >= 300] = 1


		"""
		Zoom the ct stack to make the voxels equal in size in all dimetions
		For each region cut out a (size, size, size) chunk about the nodule
		"""
		scaling_factor = z_res / xy_res
		size = 64 # shape of zyx
		input_st = np.zeros((1, size, size, size,2), dtype=np.float32) 
		output = np.zeros((9), dtype = np.float32)


		tot = zoom(tot, zoom = [scaling_factor,1,1], order = 0)
		input_im = zoom(input_im, zoom = [scaling_factor,1,1], order = 1)

		# regions
		ltot = label(tot)
		nod_nr = 0
		for reg_l in regionprops(ltot, intensity_image = input_im):

			# find malignancy
			mal = []
			coor = reg_l.centroid
			for i in range(output_mal.shape[0]):
				tmp = output_mal[i, int(coor[0]/scaling_factor), int(coor[1]), int(coor[2])]
				if tmp:
					mal.append(output_mal[i, int(coor[0]/scaling_factor), int(coor[1]), int(coor[2])])

			# if no doctors have anoted the malignancy, continue
			if not mal:
				continue
			avr_mal = sum(mal)/ len(mal)

			#round to nearest .5
			avr_mal = np.round(avr_mal * 2) / 2


			#one hot
			output = avr_mal/mask
			output[output == 1] = 1
			output[output != 1] = 0

			center = np.round(reg_l.centroid).astype(int)

			#ct
			tmp = input_im[center[0]-int(size/2):center[0]+int(size/2), center[1]-int(size/2):center[1]+int(size/2), center[2]-int(size/2):center[2]+int(size/2)]
			input_st[0, :tmp.shape[0], :tmp.shape[1], :tmp.shape[2], 0] = tmp
			
			#cut the ct images intensities to [-1024,400] then scale to [0,1]
			input_st[:, :, :, :, 0][input_st[:, :, :, :, 0] < -1024] = -1024
			input_st[:, :, :, :, 0][input_st[:, :, :, :, 0] > 400] = 400
			input_st[:, :, :, :, 0] = input_st[:, :, :, :, 0] - np.amin(input_st[:, :, :, :, 0])
			input_st[:, :, :, :, 0] = input_st[:, :, :, :, 0] / np.amax(input_st[:, :, :, :, 0])


			# remove all tumors exept the one in this region
			tot_tmp = tot.copy()
			tot_tmp[ltot != reg_l.label] = 0
			
			#binary
			tmp = tot_tmp[center[0]-int(size/2):center[0]+int(size/2), center[1]-int(size/2):center[1]+int(size/2), center[2]-int(size/2):center[2]+int(size/2)]
			input_st[0, :tmp.shape[0], :tmp.shape[1], :tmp.shape[2], 1] = tmp

			#save in end_path 
			os.makedirs(end_path, exist_ok = True)
			f = h5py.File((end_path+'/'+str(pat)+'-'+str(nod_nr) + '.hd5'), 'w')
			f.create_dataset("data", data=input_st, compression="gzip", compression_opts=4)
			f.create_dataset("label", data=output, compression="gzip", compression_opts=4)
			f.close()
			nod_nr = nod_nr + 1




end_path = '/Users/medtek/Desktop/scripts and patient data/data_for_classification_VGG_v3'
start_path = '/Users/medtek/Desktop/scripts and patient data/dataFromLIDC'

only = []
for i in range(1,1000):
	only.append(i)

classification_dataset_gen(start_path, end_path)#, only = only)
	



