import numpy as np
import tensorflow as tf
from tf_keras_vis.utils.scores import CategoricalScore
from imlib import scale_between_zero_one, scale_between_minus_one_one


def threshold_img(img, threshold_value=0.1):
    """
    Thresholds an image to a certain value.
    """
    img = scale_between_zero_one(img)
    img = tf.where(img < threshold_value, tf.zeros_like(img), img)
    img = scale_between_minus_one_one(img)
    return img


def apply_attention_on_img(img, attention_map):
    # Convert img to same pixel values [0, 1]
    img = scale_between_zero_one(img)
    attention_map_scaled = scale_between_zero_one(attention_map)
    # Interpolate by multiplication and normalise
    img = tf.math.multiply(img, attention_map_scaled)
    img = tf.math.divide(img, tf.reduce_max(img))
    return scale_between_minus_one_one(img)


def get_clf_attention_img(img, gradcam, class_index, args):
    """
    Applys gradcam to an image and returns the heatmap as well as the enhanced img.
    """
    # Make img same channel as model input
    if tf.shape(img)[-1] == 3 and args.clf_input_channel == 1:
        # Make temporary 1 channel img if 3 channel
        img_tmp = tf.image.rgb_to_grayscale(img)
    else:
        img_tmp = img

    # Generate Grad-Cam map
    cam = gradcam(CategoricalScore(class_index), img_tmp, penultimate_layer=-1)
    if np.max(cam) == 0 and np.min(cam) == 0:
        cam = tf.ones(shape=cam.shape)
    # Normalise
    cam = (cam - tf.reduce_min(cam)) / (tf.reduce_max(cam) - tf.reduce_min(cam))
    # OLD cam = tf.math.divide(cam, tf.reduce_max(cam))
    # Turn to batched channeled array and make chanel dimension compatible with img
    cam = tf.expand_dims(cam, axis=-1)
    if img.get_shape()[-1] == 3:
        cam = tf.image.grayscale_to_rgb(tf.convert_to_tensor(cam))

    # Resize to img size
    cam = scale_between_minus_one_one(cam)
    img = apply_attention_on_img(img, cam)
    return img, cam  # [-1,1]
