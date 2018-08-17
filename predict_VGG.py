import h5py
import numpy as np
from smistad.smistad_imgaug import UltrasoundImageGenerator
from smistad.smistad_dataset import get_dataset_files
import os
from tensorflow.python.keras.models import load_model


data_path = 'data_for_classification_VGG_v2'
model_path = 'auto_saved_net_classification/model_classification_3d_06_08.hd5'
test_set = []
full_set = []
val_set = []
for file in os.listdir(data_path):
	if int(file.split('-')[0]) <= 100:
		test_set.append(file.split('.')[0])
	full_set.append(file.split('.')[0])
	if int(file.split('-')[0]) > 100 and int(file.split('-')[0]) < 200:
		val_set.append(file.split('.')[0])



model = load_model(model_path, compile=False)
mask = np.arange(1, 5.5, 0.5)
gen = UltrasoundImageGenerator(get_dataset_files(data_path, only =  test_set))
cnt = 0
list_ = []
for input, output in gen.flow(batch_size = 1, shuffle = False):
    # input[:, :, :, :, 0][input[:, :, :, :, 0] < -1024] = -1024
    # input[:, :, :, :, 0][input[:, :, :, :, 0] > 400] = 400
    # input[:, :, :, :, 0] = input[:, :, :, :, 0] - np.amin(input[:, :, :, :, 0])
    # input[:, :, :, :, 0] = input[:, :, :, :, 0] / (np.amax(input[:, :, :, :, 0]) + 10e-6)
    # input_im = np.zeros((64,64,64,2)) * (-1024)
    # input_im[:input.shape[0], :input.shape[1], :input.shape[2], :input.shape[3]] = input
    #print(np.amin(input), np.amax(input))
    #print(input.shape)
    # pred = model.predict(input)
    # print(pred)



	pred = model.predict(input)
	p_max = np.amax(pred)
	pred[pred < p_max] = 0
	pred[pred != 0] = 1
	pred = pred*mask
	pred = np.unique(pred)[1]
	output = output*mask
	output = np.unique(output)[1]
	print(pred, output)
	list_.append(np.abs(output-pred))
	


	cnt = cnt + 1
	if cnt > len(test_set):
		break
print(list_)
print()
print(np.mean(list_))
print(np.std(list_))
