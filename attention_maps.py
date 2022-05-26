import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tf_keras_vis.utils.scores import CategoricalScore

from imlib import scale_to_zero_one, scale_to_minus_one_one, plot_any_img


def shift_values_above_intensity(cam, attention_intensity: float):
    """
    makes sure all values are above attention_intensity value.
    """
    cam += attention_intensity
    cam /= np.max(cam)
    return cam


def apply_gradcam(img, gradcam, class_index, attention_type, attention_intensity=1):
    """
    Applys gradcam to an image and returns the heatmap as well as the enhanced img.
    """
    # Generate cam map
    cam = gradcam(CategoricalScore(class_index), img)  # returns ndarray in [0,1]
    cam1 = gradcam(CategoricalScore(class_index), img, penultimate_layer=-10)
    cam2 = gradcam(CategoricalScore(class_index), img, penultimate_layer=-30)
    cam3 = gradcam(CategoricalScore(class_index), img, penultimate_layer=-35)
    cam4 = gradcam(CategoricalScore(class_index), img, penultimate_layer=-40)
    cam5 = gradcam(CategoricalScore(class_index), img, penultimate_layer=-45)
    cam6 = gradcam(CategoricalScore(class_index), img, penultimate_layer=-50)
    if np.max(cam) == 0 and np.min(cam) == 0:
        print(f"Found image without attention... Class index {class_index}")
        cam = tf.ones(shape=cam.shape)
    plot_any_img(img)
    plot_any_img(cam)
    plot_any_img(cam1)
    plot_any_img(cam2)
    plot_any_img(cam3)
    plot_any_img(cam4)
    plot_any_img(cam5)
    plot_any_img(cam6)

    # Turn to batched 3-channel array
    cam = tf.expand_dims(cam, axis=-1)
    cam = tf.image.grayscale_to_rgb(tf.convert_to_tensor(cam))
    # Convert img to same pixel values [0, 1]
    img = scale_to_zero_one(img)

    if attention_type == "spa-gan":
        cam = shift_values_above_intensity(cam, attention_intensity)
    if attention_intensity == 0:  # for testing purposes when you want to apply attention everywhere.
        cam = tf.ones(shape=cam.shape)

    # Interpolate by multiplication and normalise
    img = cam * img
    img /= np.max(img)
    return scale_to_minus_one_one(img), scale_to_minus_one_one(cam)  # [-1,1]