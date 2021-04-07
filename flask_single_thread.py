"""
Resnet50 classification with processing time outputs and a flask server.
Single thread to formulate the batch.
Use a for loop to preprocess image inputs and form input batch.
    for n in range(batch_size)
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

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

batch_size = 8 
app = Flask(__name__)
img_path = 'elephant.jpg'

model = ResNet50(weights='imagenet')

for n in range(10):
    batched_input = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)

    tic = time.time()
    for i in range(batch_size):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        batched_input[i, :] = x
    batched_input = tf.constant(batched_input)
    print("batch_prepare(): %s ms" % ((time.time() - tic) * 1000))
    
    tic = time.time()
    preds = model.predict(batched_input)
    print("predict(): %s ms" % ((time.time() - tic) * 1000))
   
    tic = time.time()
    results = decode_predictions(preds)
    print("decode_predictions(): %s ms \n" % ((time.time() - tic) * 1000))

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
    print("Done")
    #    app.run('0.0.0.0')
