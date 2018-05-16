#!/usr/bin/python
# -*- coding: utf-8 -*-

# Importing the Keras libraries and packages

import constants

from keras.models import Sequential
from keras.models import model_from_json

from keras.preprocessing import image

from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

# Initializing the CNN

model = Sequential()

# Convolution layer with 32 inputs

model.add(Conv2D(32, (3, 3), input_shape=(constants.img_size, constants.img_size, 3),
          activation='relu'))

# Max pooling

model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolution layer with 64 inputs

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten

model.add(Flatten())

# Full connection

model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Fit model to images

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2,
                                   zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(constants.training_set,
        target_size=(constants.img_size, constants.img_size), batch_size=constants.batch_size,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(constants.test_set,
        target_size=(constants.img_size, constants.img_size), batch_size=constants.batch_size,
        class_mode='binary')

model.fit_generator(train_generator, steps_per_epoch=constants.steps_per_epoch,
                    epochs=constants.epochs,
                    validation_data=validation_generator,
                    validation_steps=constants.validation_steps)

# Serialize Model to JSON

json = model.to_json()
with open(constants.model_json, 'w') as json_file:
    json_file.write(json)

# Serialize Weights to HDF5

model.save_weights(constants.model_weight)
