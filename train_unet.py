# testing unet3D
from math import ceil
from tensorflow.python.keras._impl.keras.callbacks import ModelCheckpoint
from smistad.smistad_imgaug import UltrasoundImageGenerator
from smistad.smistad_dataset import get_dataset_files
from smistad.smistad_network import Unet
from u_network import *
from model_eval import *
from batch_generator3 import *

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

data_path = 'dataset_3d_2_doc_agree'
save_model_path = 'saved_models'
history_path = 'Training_history'

# valdidation set
val_set_ = [81,200]
val_set = []
for i in range(val_set_[0],val_set_[1] +1):
	#patNr = '{:04d}'.format(i)
	val_set.append(i)

# save a number of images for testing 
test_set_ = [1,80]
test_set=[]
for i in range(test_set_[0],test_set_[1]+1):
	#test = '{:04d}'.format(i)
	test_set.append(i)

network = Unet(input_shape=(64, 256, 256, 1), nb_classes=2)
network.encoder_spatial_dropout = 0.1
network.decoder_spatial_dropout = 0.1
#network.set_convolutions([16, 64, 128, 256, 256, 256, 128, 64, 16])

model = network.create()

#print(model.summary())
#print('depth:',network.get_depth())

#load model
#model.load_weights('saved_models/model_3d.hd5', by_name=True)

model.compile(
	optimizer='adadelta',
	loss=network.get_dice_loss(),
)

save_best = ModelCheckpoint(
	save_model_path+'/model_3d.hd5',
	monitor='val_loss',
	verbose=0,
	save_best_only=True,
	save_weights_only=False,
	mode='auto',
	period=1
)

#augmentation
#aug = {'rotate': 20,'flip':1}
aug = {'flip':1}
batch_size = 1
epochs = 200

train_length = batch_length(get_dataset_files(data_path, exclude = val_set + test_set))
val_length = batch_length(get_dataset_files(data_path, only = val_set))

train_gen = batch_gen3(get_dataset_files(data_path, exclude = val_set + test_set), batch_size, aug, shuffle_list = True, epochs = epochs)  
val_gen = batch_gen3(get_dataset_files(data_path, only = val_set), batch_size, shuffle_list = True, epochs = epochs)

history = model.fit_generator(
	train_gen,
	steps_per_epoch =ceil(train_length/batch_size),
	epochs = epochs,
	validation_data=val_gen,
	validation_steps=ceil(val_length/batch_size),
	callbacks = [save_best]
)


# save history:
f = h5py.File((history_path+'/over_the_weekend.hd5'), 'w')
f.create_dataset("history", data=history, compression="gzip", compression_opts=4)
f.close()




