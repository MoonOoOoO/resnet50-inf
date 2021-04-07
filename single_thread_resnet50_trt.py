"""
Resent50 classification
One image at a time. Output processing time.
"""
import time
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from flask import Flask, request
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.saved_model import tag_constants

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

app = Flask(__name__)

model = tf.saved_model.load('trt-resnet', tags=[tag_constants.SERVING])
infer = model.signatures['serving_default']

# model = ResNet50(weights='imagenet')

img_path = 'elephant.jpg'

def test_predict_process():
    tic = time.time()
    img = image.load_img(img_path, target_size=(224, 224))
    print("image.load_image(): %s ms" % ((time.time() - tic) * 1000))

    tic = time.time()
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = tf.constant(x)
    print("preprocess(): %s ms" % ((time.time() - tic) * 1000))

    tic = time.time()
    labeling = infer(x)
    print(labeling)
    #pre = labeling['probs'].numpy()
    #pre = model.predict(x)
    print("model.predict(): %s ms" % ((time.time() - tic) * 1000))

    #tic = time.time()
    #results = decode_predictions(pre)[0]
    #print("decode_predictions(): %s ms" % ((time.time() - tic) * 1000))

def prepare_image(raw_image):
    img = cv2.resize(raw_image, dsize=(224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

@app.route('/')
def index():
    return "hello world"

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == 'POST':
        image_file = request.files['file'].read()
        if image_file:
            image_array = np.fromstring(image_file, np.uint8)
            image_decode = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            x = prepare_image(image_decode)
            predicts = model.predict(x)
            results = decode_predictions(predicts)
            return str(results)

if __name__ == '__main__':
    time.sleep(2)
    for i in range(10):
        print("\n")
        test_predict_process()
    #    app.run('0.0.0.0')

