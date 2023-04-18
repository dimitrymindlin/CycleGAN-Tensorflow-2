import numpy as np
import tensorflow as tf
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


def get_clf_attention_img(img, gradcam, class_index, attention_type, attention_intensity=1, attention_source="clf"):
    """
    CURRENTLY ONLY GRADCAM IS SUPPORTED
    Applys gradcam to an image and returns the heatmap as well as the enhanced img.
    """
    # Generate cam map
    img_tmp = img
    # measure time to compute cam
    cam = gradcam(CategoricalScore(class_index), img_tmp, penultimate_layer=-1)
    if np.max(cam) == 0 and np.min(cam) == 0:
        cam = tf.ones(shape=cam.shape)
    # Normalise cam map
    cam = tf.math.divide(cam, tf.reduce_max(cam))
    # Threshold low values
    cam = threshold_img(cam, threshold_value=0.1)

    # Dilate the cam map to indluce more context of the img
    # Define the size of the neighborhood
    neighborhood_size = 20
    from scipy.ndimage.filters import maximum_filter
    # Perform dilation on the tensor
    # Define a wrapper function to apply maximum_filter to the input tensor
    def apply_maximum_filter(x):
        return maximum_filter(x, size=neighborhood_size)

    dilated_arr = tf.numpy_function(func=apply_maximum_filter, inp=[cam], Tout=tf.float32)

    # Turn to batched channeled array and make compatible with img
    cam = tf.expand_dims(dilated_arr, axis=-1)

    if img.get_shape()[-2] == 256 and attention_source != "discriminator":
        cam = tf.image.resize(cam, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    if attention_type == "spa-gan":
        cam = shift_values_above_intensity(cam, attention_intensity)
    if attention_intensity == 0:  # for testing purposes when you want to apply attention everywhere.
        cam = tf.ones(shape=cam.shape)
    cam = scale_between_minus_one_one(cam)
    img = apply_attention_on_img(img, cam)
    return img, cam  # [-1,1]


"""def apply_lime(img, model, class_index):
    # Applies LIME XAI Method to the image to obtain the attention map.
    # Define explainer
    explainer = lime_image.LimeImageExplainer()

    # Get explanations for both classes
    explanation = explainer.explain_instance(tf.squeeze(img).numpy(), model.predict, top_labels=2, hide_color=0,
                                             num_samples=1000)

    # Get attention map and image for class_index
    attention_map, image = explanation.get_image_and_mask(class_index, positive_only=True,
                                                          num_features=5, hide_rest=True)
    plot_any_img(attention_map)
    plot_any_img(img)
    return attention_map, image"""


def add_attention_maps_to_single_ds(dataset, gradcam, label_index, img_height, img_width):
    attention_maps = []

    # normalizing the images to [-1, 1]
    def normalize(image):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [img_height, img_width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = (image / 127.5) - 1
        return image

    for img, _ in dataset:
        img_tmp = np.copy(img)
        img_tmp = np.expand_dims(normalize(img_tmp), axis=0)
        img_tmp = tf.convert_to_tensor(img_tmp)
        _, cam = get_clf_attention_img(img_tmp, gradcam, label_index,
                                       "attention-gan-original",
                                       attention_intensity=1,
                                       attention_source="clf")
        # remove batch dimension (first)
        cam = tf.squeeze(cam, axis=0)
        attention_maps.append(cam)
    # Turn list to tensor slices
    attention_maps = tf.convert_to_tensor(attention_maps)
    attention_ds = tf.data.Dataset.from_tensor_slices(attention_maps)
    # zip dataset and attention_ds
    dataset = tf.data.Dataset.zip((dataset, attention_ds))
    return dataset


def add_attention_maps(A_train, B_train, gradcam, img_height, img_width):
    A_train = add_attention_maps_to_single_ds(A_train, gradcam, 0, img_height, img_width)
    B_train = add_attention_maps_to_single_ds(B_train, gradcam, 1, img_height, img_width)
    return A_train, B_train
