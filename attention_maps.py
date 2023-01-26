import numpy as np
import tensorflow as tf
from lime import lime_image
from tf_keras_vis.utils.scores import CategoricalScore

from imlib import scale_between_zero_one, scale_between_minus_one_one, plot_any_img


def shift_values_above_intensity(img, attention_intensity: float):
    """
    makes sure all values are above attention_intensity value.
    """
    img += attention_intensity
    img = tf.math.divide(img, tf.reduce_max(img))
    return img


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


def apply_gradcam(img, gradcam, class_index, args, attention_intensity=1, attention_source="clf"):
    """
    Applys gradcam to an image and returns the heatmap as well as the enhanced img.
    """
    # Generate cam map
    if tf.shape(img)[-1] == 3 and args.clf_input_channel == 1:
        # Make temporary 1 channel img if 3 channel
        img_tmp = tf.image.rgb_to_grayscale(img)
    else:
        img_tmp = img

    if img.get_shape()[-2] == 256 and attention_source != "discriminator":
        cam_input = tf.image.resize(img_tmp, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        cam = gradcam(CategoricalScore(class_index), cam_input, penultimate_layer=-1)
    else:
        cam = gradcam(CategoricalScore(class_index), img_tmp, penultimate_layer=-1)
    if np.max(cam) == 0 and np.min(cam) == 0:
        cam = tf.ones(shape=cam.shape)
    cam = tf.math.divide(cam, tf.reduce_max(cam))

    # Turn to batched channeled array and make compatible with img
    cam = tf.expand_dims(cam, axis=-1)
    if img.get_shape()[-1] == 3:
        cam = tf.image.grayscale_to_rgb(tf.convert_to_tensor(cam))
    if img.get_shape()[-2] == 256 and attention_source != "discriminator":
        cam = tf.image.resize(cam, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    if args.attention_type == "spa-gan":
        cam = shift_values_above_intensity(cam, attention_intensity)
    if attention_intensity == 0:  # for testing purposes when you want to apply attention everywhere.
        cam = tf.ones(shape=cam.shape)
    cam = scale_between_minus_one_one(cam)
    img = apply_attention_on_img(img, cam)
    return img, cam  # [-1,1]


def apply_lime(img, model, class_index):
    """Applies LIME XAI Method to the image to obtain the attention map."""
    # Define explainer
    explainer = lime_image.LimeImageExplainer()

    # Get explanations for both classes
    explanation = explainer.explain_instance(tf.squeeze(img).numpy(), model.predict, top_labels=2, hide_color=0, num_samples=1000)

    # Get attention map and image for class_index
    attention_map, image = explanation.get_image_and_mask(class_index, positive_only=True,
                                                          num_features=5, hide_rest=True)
    plot_any_img(attention_map)
    plot_any_img(img)
    return attention_map, image
