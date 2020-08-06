from flask import Flask, render_template, request, jsonify
from werkzeug import secure_filename
from keras.models import load_model
import cv2
import numpy as np

import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

@app.route('/uploaderkatarak', methods=['GET', 'POST'])
def upload_katarak():
    if request.method == 'POST':
        file = request.form['file']
        starter = file.find(',')
        image_data = file[starter + 1:]
        image_data = bytes(image_data, encoding="ascii")
        im = Image.open(BytesIO(base64.b64decode(image_data)))
        im.save('images/image.jpg')

        my_image = im
        my_label_y = [1]

        fname = "images/image_katarak.jpg"
        model = load_model('katarak_model.h5')
        # model = load_weights('model.h5')
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        # Read the image
        img = cv2.imread(fname)
        img = cv2.resize(img,(128,128))
        img = np.reshape(img,[1,3,128,128])

        my_predicted_image = model.predict_classes(img)
    return  jsonify(result=my_predicted_image)

@app.route('/uploaderpterigium', methods=['GET', 'POST'])
def upload_pterigium():
    if request.method == 'POST':
        file = request.form['file']
        starter = file.find(',')
        image_data = file[starter + 1:]
        image_data = bytes(image_data, encoding="ascii")
        im = Image.open(BytesIO(base64.b64decode(image_data)))
        im.save('images/pterigium.jpg')

        my_image = im
        my_label_y = [1]

        fname = "images/image_pterigium.jpg"
        model = load_model('pterigium_model.h5')
        # model = load_weights('model.h5')
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        # Read the image
        img = cv2.imread(fname)
        img = cv2.resize(img,(128,128))
        img = np.reshape(img,[1,3,128,128])

        my_predicted_image = model.predict_classes(img)
    return  jsonify(result=my_predicted_image)

if __name__ == '__main__':
    app.debug = True
    app.run(host = '0.0.0.0',port=80)