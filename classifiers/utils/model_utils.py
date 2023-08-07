import tensorflow as tf


def get_model_by_name(config, input_shape, weights, img_input=None):
    base_model = tf.keras.applications.InceptionV3(include_top=False,
                                                   input_shape=input_shape,
                                                   input_tensor=img_input,
                                                   weights=weights,
                                                   pooling=config['model']['pooling'],
                                                   classes=len(config['data']['class_names']))
    return base_model


def get_input_shape_from_config(config):
    return (
        config['data']['image_height'],
        config['data']['image_width'],
        config['data']['image_channel']
    )
