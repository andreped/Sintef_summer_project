import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.cElementTree as et
from sklearn import metrics
from skimage.measure import regionprops, label
import h5py


# takes in ground truth (output), and pred_output (predicted)
# retuns an array of the scores [specificity, sensitivity_3d, sensitivity_2d, diceScore, dice_TP]
# if there are no tumors, returns -1
def all_scores(output, pred_output):
	if np.count_nonzero(output[:,:,:,1]) == 0:
		return -1

	list_ =[specificity(output, pred_output.copy()), 
			sensitivity_3d(output, pred_output.copy()), 
			sensitivity_2d(output, pred_output.copy()), 
			diceScore(output, pred_output.copy()), 
			dice_TP(output, pred_output.copy())]
	return list_

	


'''
######################################################################################
					---different measurements----
- dice_score:
-> calculates the total intersection over union of the predicted output and the output (ground truth)

- dice_TP:
-> calculates the intersection over union of the true positives

- specifisity:
-> calculates how may of the predicted tumors are false positives, in 3d

- sensitivity_2d:
-> calculates the number of true positives, per slice. 'how many of the tumors did the algorithm find in that slice'

- sensitivity_3d:
-> calculates the number of true positives, in the whole voulme

--- all functions expect binary inputs -----


#######
'''

def specificity(output, pred_output):
	if np.count_nonzero(output[:,:,:,1]) == 0:
		return -1

	labels = label(pred_output[:, :, :, 1])
	region = regionprops(labels)

	CM = []
	for reg in region:
		CM.append(reg.centroid)

	TP = 0
	for i in range(len(CM)):
		if output[int(CM[i][0]), int(CM[i][1]), int(CM[i][2]), 1] != 0:
			TP = TP + 1
	if not CM:
		return 0
	FN = len(CM) - TP
	spec = TP/(TP + FN) * 100
	return spec

def sensitivity_3d(output, pred_output):
	if np.count_nonzero(output[:, :, :, 1]) == 0:
		return -1

	#print('sense 3d:', pred_output.shape, output.shape)

	inter = np.multiply(output[:, :, :, 1], pred_output[:output.shape[0], :, :, 1])
	labels_output = label(output[:, :, :, 1])
	labels_inter = label(inter)
	sens_3d = (len(np.unique(labels_inter)) - 1) / (len(np.unique(labels_output)) - 1)
	return (sens_3d * 100)

def sensitivity_2d(output, pred_output):
	if np.count_nonzero(output[:,:,:,1]) == 0:
		return -1

	list_ = []
	for i in range(output.shape[0]):
		labels = label(output[i, :, :, 1])
		if np.count_nonzero(labels) == 0:
			continue
		list_.append((len(np.unique(np.multiply(labels,pred_output[i, :, :, 1])))-1) / (len(np.unique(labels))-1))

	sens = (sum(list_) / len(list_)) * 100
	return sens

def diceScore(output, pred_output):
	if np.count_nonzero(output[:,:,:,1]) == 0:
		return -1

	pad_output = np.zeros(pred_output.shape)
	for i in range(output.shape[0]):
		pad_output[i,:,:,:] = output[i,:,:,:]

	intersection  = np.count_nonzero(np.multiply(pad_output[:,:,:,1],pred_output[:,:,:,1])) * 2
	union = np.count_nonzero(pad_output[:,:,:,1]) + np.count_nonzero(pred_output[:,:,:,1])
	dice_score = (intersection/union) * 100
	return dice_score

def dice_TP(output, pred_output):
	if np.count_nonzero(output[:,:,:,1]) == 0:
		return -1

	pred_output = pred_output[:output.shape[0],:,:,:]
	inter = np.multiply(output[:, :, :, 1], pred_output[:, :, :, 1])
	labels_pred = label(pred_output[:, :, :, 1])
	labels_left = np.multiply(inter,labels_pred)
	left = list(np.unique(labels_left))
	tot = list(np.unique(labels_pred))
	for i in tot:
		if not i in left:
			pred_output[:,:,:,1][labels_pred == i] = 0


	pad_output = np.zeros(pred_output.shape)
	for i in range(output.shape[0]):
		pad_output[i,:,:,:] = output[i,:,:,:]

	intersection  = np.count_nonzero(np.multiply(pad_output[:,:,:,1],pred_output[:,:,:,1])) * 2
	union = np.count_nonzero(pad_output[:,:,:,1]) + np.count_nonzero(pred_output[:,:,:,1])
	dice_score_TP = (intersection/union) * 100
	return dice_score_TP

######################################################################################



#saves scores in 'saved_scores' with the model id as sub-directory
def save_score(score, model_id, pat_nr):
	score = np.asarray(score)
	path = 'saved_scores'
	os.makedirs(path+'/'+model_id, exist_ok = True)	
	f = h5py.File((path+'/'+model_id+'/'+str(pat_nr) + '.hd5'), 'w')
	f.create_dataset("score", data=score, compression="gzip", compression_opts=4)
	f.close()
	print('saved score for patient nr:',pat_nr)

