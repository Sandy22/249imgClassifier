#!/usr/bin/python
# -*- coding: utf-8 -*-

# Importing the Keras libraries and packages

#Start the parameter server
#  python classify.py --job_name="ps" --task_index=0
  
#Start the three workers
 # python classify.py --job_name="worker" --task_index=0
 # python classify.py --job_name="worker" --task_index=1
 # python classify.py --job_name="worker" --task_index=2

import helper
import argparse
import numpy as np

from keras.models import Sequential
from keras.models import model_from_json

from keras.preprocessing import image

from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import keras

from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

cluster = tf.train.ClusterSpec({"ps": ["localhost:2222"],
"worker": ["localhost:2223"]})

server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)

batch_size = 32
training_set = 'dataset/train'
test_set = 'dataset/test'
epochs = 50
steps_per_epoch = helper.stepCount(training_set,batch_size)
validation_steps = helper.stepCount(test_set,batch_size)
img_size = 64

if FLAGS.job_name == "ps":
	server.join()
elif FLAGS.job_name == "worker":


    # Assign operations to local server
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(img_size, img_size, 3),activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(units=128, activation='relu'))
	model.add(Dense(units=1, activation='sigmoid'))
	model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

	train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2,
                                   zoom_range=0.2, horizontal_flip=True)

	test_datagen = ImageDataGenerator(rescale=1. / 255)

	train_generator = train_datagen.flow_from_directory(training_set,
        	target_size=(img_size, img_size), batch_size=batch_size,
       		class_mode='binary')

	validation_generator = test_datagen.flow_from_directory(test_set,
        	target_size=(img_size, img_size), batch_size=batch_size,
        	class_mode='binary')


   	with tf.device(tf.train.replica_device_setter(
		worker_device="/job:worker/task:%d" % FLAGS.task_index,
		cluster=cluster)):
		print("done")

		global_step = tf.contrib.framework.get_or_create_global_step()


		with tf.Session(server.target) as mon_sess:

			model.fit_generator(train_generator, 
		    	steps_per_epoch=steps_per_epoch,
                    	epochs=epochs,
                    	validation_data=validation_generator,
                    	validation_steps=validation_steps)
			model_json = model.to_json()
			with open('model.json', 'w') as json_file:
				json_file.write(model_json)

			model.save_weights('model.h5')
	
			print("done")
			mon_sess.close()
		
