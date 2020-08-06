from flask import Flask, render_template, request, jsonify
from werkzeug import secure_filename
from apps import *

import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.form['file']
        starter = file.find(',')
        image_data = file[starter + 1:]
        image_data = bytes(image_data, encoding="ascii")
        im = Image.open(BytesIO(base64.b64decode(image_data)))
        im.save('images/image.jpg')

        my_image = im
        my_label_y = [1]

        fname = "images/image.jpg"
        image = np.array(ndimage.imread(fname, flatten=False))
        my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((num_px * num_px * 3, 1))
        my_image = my_image / 255.
        my_predicted_image = predict(my_image, my_label_y, parameters)

    return  classes[int(np.squeeze(my_predicted_image)),].decode("utf-8"), 201

if __name__ == '__main__':
    app.debug = True
    app.run(host = '0.0.0.0',port=80)