import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tf_keras_vis.utils.scores import CategoricalScore

from imlib import scale_between_zero_one, scale_between_minus_one_one, plot_any_img


def shift_values_above_intensity(cam, attention_intensity: float):
    """
    makes sure all values are above attention_intensity value.
    """
    cam += attention_intensity
    cam = tf.math.divide(cam, tf.reduce_max(cam))
    return cam


def apply_attention_on_img(img, attention_map):
    # Convert img to same pixel values [0, 1]
    img = scale_between_zero_one(img)
    attention_map_scaled = scale_between_zero_one(attention_map)
    # Interpolate by multiplication and normalise
    img = tf.math.multiply(img, attention_map_scaled)
    img = tf.math.divide(img, tf.reduce_max(img))
    return scale_between_minus_one_one(img)


def apply_gradcam(img, gradcam, class_index, attention_type, attention_intensity=1, attention_source="clf"):
    """
    Applys gradcam to an image and returns the heatmap as well as the enhanced img.
    """
    # Generate cam map
    if img.get_shape()[-2] == 256 and attention_source != "discriminator":
        cam_input = tf.image.resize(img, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        cam = gradcam(CategoricalScore(class_index), cam_input, penultimate_layer=-1)
    else:
        cam = gradcam(CategoricalScore(class_index), img, penultimate_layer=-1)
    if np.max(cam) == 0 and np.min(cam) == 0:
        print(f"Found image without attention...")
        cam = tf.ones(shape=cam.shape)
    cam = tf.math.divide(cam, tf.reduce_max(cam))
    # Turn to batched 3-channel array
    cam = tf.expand_dims(cam, axis=-1)
    cam = tf.image.grayscale_to_rgb(tf.convert_to_tensor(cam))
    if img.get_shape()[-2] == 256 and attention_source != "discriminator":
        cam = tf.image.resize(cam, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    if attention_type == "spa-gan":
        cam = shift_values_above_intensity(cam, attention_intensity)
    if attention_intensity == 0:  # for testing purposes when you want to apply attention everywhere.
        cam = tf.ones(shape=cam.shape)
    cam = scale_between_minus_one_one(cam)
    img = apply_attention_on_img(img, cam)
    return img, cam  # [-1,1]
