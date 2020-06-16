
import tensorflow as tf
import tensorflow_hub as hub
import json


IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

def get_class_names(json_file):
    with open(json_file, 'r') as f:
        class_names = json.load(f)
    class_names_2 = dict()
    for num in class_names:
        class_names_2[str(int(num)-1)] = class_names[num] #create class name array
    return class_names_2


def load_model(model_folder):
    reloaded_model = tf.keras.models.load_model(model_folder,custom_objects={'KerasLayer':hub.KerasLayer})
    print(reloaded_model.summary())
    return reloaded_model

def process_image(numpy_image):
    print(numpy_image.shape)
    tensor_img = tf.image.convert_image_dtype(numpy_image, dtype=tf.int16, saturate=False)
    resized_img = tf.image.resize(numpy_image,(IMG_SIZE,IMG_SIZE)).numpy()
    norm_img = resized_img/255
    return norm_img