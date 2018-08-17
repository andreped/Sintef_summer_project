# conv net classification


from smistad.VGGnet_3d import *
from math import ceil
from tensorflow.python.keras._impl.keras.callbacks import ModelCheckpoint
from smistad.smistad_imgaug import UltrasoundImageGenerator
from smistad.smistad_dataset import get_dataset_files
import os
import matplotlib.pyplot as plt

data_path = 'data_for_classification_VGG_v2'
save_model_path = 'auto_saved_net_classification'
test_set = []
val_set = []
for file in os.listdir(data_path):
	if int(file.split('-')[0]) <= 100:
		test_set.append(file.split('.')[0])
	if int(file.split('-')[0]) > 100 and int(file.split('-')[0]) < 200:
		val_set.append(file.split('.')[0])
#print(len(test_set))
#print(len(val_set))



network = VGGnet_3d((64,64,64,2),9)
network.set_dense_size(100)
network.encoder_spatial_dropout = 0.3

model = network.create()
print(model.summary())

#load model
# model.load_weights('auto_saved_net_classification/model_classification_3d.hd5', by_name=True)

model.compile(
	optimizer='adam',
	loss='categorical_crossentropy',
)

save_best = ModelCheckpoint(
	save_model_path+'/model_classification_3d.hd5',
	monitor='val_loss',
	verbose=0,
	save_best_only=True,
	save_weights_only=False,
	mode='auto',
	period=1
)



training_generator = UltrasoundImageGenerator(get_dataset_files(data_path, exclude = val_set+test_set))
training_generator.add_flip_3d()
training_generator.add_rotate_3d(-40,40)
validation_generator = UltrasoundImageGenerator(get_dataset_files(data_path, only =  val_set))



batch_size = 64
history = model.fit_generator(
	training_generator.flow(batch_size=batch_size),
	steps_per_epoch = ceil(training_generator.get_size()/batch_size),
	epochs = 300,
	validation_data=validation_generator.flow(batch_size=batch_size),
	validation_steps=ceil(validation_generator.get_size()/batch_size),
	callbacks = [save_best]
)
