import keras.applications
import keras
import numpy as np
from keras.models import Model
from keras.utils import img_to_array


def get_model():
    vgg_model = keras.applications.VGG16(include_top=True, weights='imagenet')
    vgg_model.layers.pop()
    vgg_model.layers.pop()
    x = vgg_model.input
    y = vgg_model.output

    return Model(x, y)


def get_features(model: Model, img):
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.vgg16.preprocess_input(x)
    return model.predict(x)
