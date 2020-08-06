from flask import Flask, render_template, request, jsonify, json
from werkzeug import secure_filename
from keras.models import load_model
import keras

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
        random=np.random.rand()
        image_katarak="images/image_katarak"+str(random)+".jpg"
        im.save(image_katarak)

        my_image = im
        my_label_y = [1]

        fname = image_katarak
        model = keras.models.load_model('katarak_model.h5')
        # model = load_weights('model.h5')
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        # Read the image
        img = cv2.imread(fname)
        h,w = img.shape[:2]
        if(h>w):
            img = img[0:w, 0:w]
        else:
            img = img[0:h, 0:h]
        img = cv2.resize(img,(128,128))
        img = np.reshape(img,[1,3,128,128])

        my_predicted_image = model.predict_classes(img).tolist()
        my_acc_image = model.model.predict_proba(img).tolist()
        data = {'result': str (my_predicted_image[0]), 'acc': str (my_acc_image[0][my_predicted_image[0]])}
        #data['result']=my_predicted_image
        response = app.response_class(response=json.dumps(data),
                                  status=200,
                                  mimetype='application/json')
    return response

@app.route('/uploaderpterigium', methods=['GET', 'POST'])
def upload_pterigium():
    if request.method == 'POST':
        file = request.form['file']
        starter = file.find(',')
        image_data = file[starter + 1:]
        image_data = bytes(image_data, encoding="ascii")
        random=np.random.rand()
        image_pterigium="images/image_pterigium"+str(random)+".jpg"
        im = Image.open(BytesIO(base64.b64decode(image_data)))
        im.save(image_pterigium)

        my_image = im
        my_label_y = [1]

        fname = image_pterigium
        model = keras.models.load_model('pterigium_model.h5')
        # model = load_weights('model.h5')
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        # Read the image
        img = cv2.imread(fname)
        h,w = img.shape[:2]
        if(h>w):
            img = img[0:w, 0:w]
        else:
            img = img[0:h, 0:h]
        img = cv2.resize(img,(128,128))
        img = np.reshape(img,[1,3,128,128])

        my_predicted_image = model.predict_classes(img).tolist()
        my_acc_image = model.model.predict_proba(img).tolist()
        data = {'result': str (my_predicted_image[0]), 'acc': str (my_acc_image[0][my_predicted_image[0]])}
        #data['result']=my_predicted_image
        response = app.response_class(response=json.dumps(data),
                                  status=200,
                                  mimetype='application/json')
    return response

if __name__ == '__main__':
    app.debug = True
    app.run(host = '0.0.0.0',port=80)