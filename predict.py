#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
from flask import Flask, render_template

import numpy as np
import constants

from keras.preprocessing import image
from keras.models import model_from_json

app = Flask(__name__)


@app.route('/')
@app.route('/index')
def show_index():
    return render_template('index.html')


@app.route('/predict/<animalName>')
def predict(animalName):

    # Load JSON and Create Model

    json_file = open(constants.model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # Load Weight into New Model

    model.load_weights(constants.model_weight)

    # Evaluate Loaded Model On Test Data

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Predicting images

    img = image.load_img(animalName, target_size=(constants.img_size,
                         constants.img_size))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    classes = model.predict(img)
    if classes[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'

    return render_template('result.html', animal=prediction)

if __name__ == '__main__':
    app.run(debug=True)
