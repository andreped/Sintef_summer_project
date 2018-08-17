
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
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn import metrics
from skimage.measure import regionprops, label


#start_path:	path to the directory that the LIDC images is stored in
#exclude:		a list of the patients that will be excluded
#only:			a list of the patients that will be exstracted
#returns: 		
#dcm_files - a dict with patient number as keys, and the path to the dicom files as value
#xml_files - a dict with patient number as keys, and the path to the xml files as value
def find_files_lidc(start_path, exclude = [], only = []):
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

	return dcm_files, xml_files

def read_dicom_images_lidc_opinions(start_path, only = [], exclude = []):
	dcm_files, xml_files = find_files_lidc(start_path, only = only)

	for pat in dcm_files.keys():
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
		#tmp = output_im.copy()

		input_im = np.zeros((len(dcm_files[pat]), 512, 512,1))
		cnt = 0
		for orig, path in sorted(orig):
			itkimage = sitk.ReadImage(path)
			ct_scan = sitk.GetArrayFromImage(itkimage)
			im = ct_scan[0,:,:]
			input_im[cnt, :, :,0] = im

			# large nodules
			for list_ in range(len(nod_list)):
				if orig in nod_list[list_].keys():

					for j in range(len(nod_list[list_][orig])):
						output_im[list_, cnt, nod_list[list_][orig][j][1], nod_list[list_][orig][j][0], 1] = 255

					output_im[list_, cnt, :, :, 1] = object_filler(output_im[list_, cnt, :, :, 1], (0,0))
					#output_im[list_, cnt, :, :, 0] = cv2.medianBlur(np.uint8(output_im[list_, cnt, :, :,0]), 7) 
			'''
			if get_small:
				# small nodules
				for list_ in range(len(nod_small_list)):
					if orig in nod_small_list[list_].keys():

						for j in range(len(nod_small_list[list_][orig])):
							output_im[list_, cnt, :, :,1] = cv2.circle(output_im[list_, cnt, :, :, 1], (nod_small_list[list_][orig][j][0], nod_small_list[list_][orig][j][1]),2,255)
							#output_im[list_, cnt, :, :, 0] = object_filler(output_im[list_, cnt, :, :, 0], (0,0))
							output_im[list_, cnt, :, :,1] = cv2.putText(output_im[list_, cnt, :, :, 1],'small nodule', (nod_small_list[list_][orig][j][0], nod_small_list[list_][orig][j][1] + 20),4, 0.5,255)
			if get_non:
			# non nodules		
				for list_ in range(len(non_nod_list)):
					if orig in non_nod_list[list_].keys():

						for j in range(len(non_nod_list[list_][orig])):
							output_im[list_, cnt, :, :,1] = cv2.circle(output_im[list_, cnt, :, :, 1], (non_nod_list[list_][orig][j][0], non_nod_list[list_][orig][j][1]),3,255)
							output_im[list_, cnt, :, :,1] = cv2.putText(output_im[list_, cnt, :, :, 1],'non nodule', (non_nod_list[list_][orig][j][0], non_nod_list[list_][orig][j][1] + 20),4,0.5,255)
			'''
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

		yield input_im, out

def save_3d(generator, end_path, out_dim = (64,256,256)):
	pat_nr = 1
	os.makedirs(end_path, exist_ok = True)
	for input_im, output in generator:
		#check the number of chunks with tumors in them
		cnt = 0
		for i in range(int(np.ceil(input_im.shape[0]/out_dim[0]))):
			if np.count_nonzero(output[i*out_dim[0]:i*out_dim[0]+out_dim[0], :, :, 1]) != 0:
				cnt = cnt+1
		output_st = np.zeros((cnt,out_dim[0],out_dim[1],out_dim[2], output.shape[-1]), dtype = np.float32)
		input_st = np.zeros((cnt,out_dim[0],out_dim[1],out_dim[2], input_im.shape[-1]), dtype = np.float32)
		
		#save the chunck 
		cnt = 0
		for i in range(int(np.ceil(input_im.shape[0]/out_dim[0]))):
			if np.count_nonzero(output[i*out_dim[0]:i*out_dim[0]+out_dim[0], :, :, 1]) == 0:
				continue
			for j in range(out_dim[0]):
				if i*out_dim[0]+j >= output.shape[0]:
					break
				output_st[cnt, j, :, :, 0] = cv2.resize(output[i*out_dim[0]+j, :, :, 0], (out_dim[1],out_dim[2]),interpolation=cv2.INTER_NEAREST)
				output_st[cnt, j, :, :, 1] = cv2.resize(output[i*out_dim[0]+j, :, :, 1], (out_dim[1],out_dim[2]),interpolation=cv2.INTER_NEAREST)
				input_st[cnt, j, :, :, 0] = cv2.resize(input_im[i*out_dim[0]+j, :, :, 0], (out_dim[1],out_dim[2]))
			cnt = cnt +1
		input_st[input_st <= -1024] = -1024
		input_st[input_st > 400] = 400


		f = h5py.File((end_path+'/'+str(pat_nr) + '.hd5'), 'w')
		f.create_dataset("data", data=input_st, compression="gzip", compression_opts=4)
		f.create_dataset("label", data=output_st, compression="gzip", compression_opts=4)
		f.close()

		print('Saved patient nr:',pat_nr)
		pat_nr = pat_nr +1


start_path = '/Users/medtek/Desktop/scripts and patient data/dataFromLIDC'
model_path = 'trained_nets/model_3d_23_07.hd5'
end_path = '/Users/medtek/Desktop/scripts and patient data/dataset_3d_2_doc_agree'
save_pred_path = '/Users/medtek/Desktop/scripts and patient data/saved_predictions'


save_3d(read_dicom_images_lidc_opinions(start_path), end_path)

