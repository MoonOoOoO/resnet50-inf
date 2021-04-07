"""
Use multithread to preprocee image intpus and formulate input batch.
Use model batch prediction.
Contains Tensorflow model and TensorRT model.
"""
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.python.saved_model import tag_constants
from concurrent.futures import ThreadPoolExecutor

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

BATCH_SIZE = 16 
img_path = 'elephant.jpg'

"""
Choose between TF model or TRT model
"""
#model = ResNet50(weights='imagenet')
model = tf.saved_model.load('trt-resnet', tags=[tag_constants.SERVING])
infer = model.signatures['serving_default']

def preprocess_image(count):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def form_batch():
    batch = np.zeros((BATCH_SIZE, 224, 224, 3), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = [executor.submit(preprocess_image, param) for param in range(BATCH_SIZE)]
        for i, x in zip(range(BATCH_SIZE), results):
            batch[i, :] = x.result()
    return tf.constant(batch)


def print_time():
    timer = time.time()
    batch = form_batch()
    print("form_batch(): %s ms" % ((time.time() - timer) * 1000))

    timer = time.time()
    labeling = infer(batch)
    # preds = labeling['probs'].numpy()
    # preds = model.predict(batch)
    print("predict(): %s ms" % ((time.time() - timer) * 1000))

    timer = time.time()
    preds = labeling['predictions'].numpy()
    decode_predictions(preds)
    print("decode_predictions(): %s ms \n" % ((time.time() - timer) * 1000))

def benchmarking():
    N_WARMUP_RUN = 50
    N_RUN = 1000
    elapsed_time = []
    batch = form_batch()

    for i in range(N_WARMUP_RUN):
        labeling = infer(batch)
        #preds = model.predict(batch)

    for i in range(N_RUN):
        start_time = time.time()
        labeling = infer(batch)
        #preds = model.predict(batch)
        end_time = time.time()
        elapsed_time = np.append(elapsed_time, end_time - start_time)
        if i % 50 == 0:
            print('Step {}: {:4.1f}ms'.format(i, (elapsed_time[-50:].mean()) * 1000))
    print('Throughput: {:.0f} images/s'.format(N_RUN * BATCH_SIZE / elapsed_time.sum()))

def previous_code():
    for n in range(10):
        batched_input = np.zeros((BATCH_SIZE, 224, 224, 3), dtype=np.float32)
        tic = time.time()
        for i in range(BATCH_SIZE):
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
        decode_predictions(preds)
        print("decode_predictions(): %s ms \n" % ((time.time() - tic) * 1000))


if __name__ == '__main__':
    # previous_code() 
    for n in range(10):
        print_time()
    benchmarking()
