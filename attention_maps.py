import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tf_keras_vis.utils.scores import CategoricalScore

from imlib import scale_to_zero_one, scale_to_minus_one_one


def get_gradcam(img, gradcam, class_index, attention_type, attention_intensity=0.5):
    # Generate cam map
    cam = gradcam(CategoricalScore(class_index), img) # returns img in [0,1]
    if np.max(cam) == 0:
        cam += 0.001
    cam /= np.max(cam)  # normalise
    # Turn to [1,512,512,3]
    cam = tf.expand_dims(cam, axis=-1)
    cam = tf.image.grayscale_to_rgb(tf.convert_to_tensor(cam))
    # Convert img to same pixel values [0, 1]
    img = scale_to_zero_one(img)

    if attention_type == "spa-gan":
        # For Spagan, enhance important parts but don't delete unimportant ones
        cam += attention_intensity
        cam /= np.max(cam)

    # Interpolate by multiplication and normalise
    img = cam * img
    img /= np.max(img)
    return scale_to_minus_one_one(img), scale_to_minus_one_one(cam)  # [-1,1]
