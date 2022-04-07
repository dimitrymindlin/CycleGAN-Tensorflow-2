import tensorflow as tf

from imlib import scale_to_zero_one, scale_to_minus_one_one


def add_background_to_img(foreground, background):
    img = tf.math.add(foreground, background)
    return tf.math.divide(img, tf.math.reduce_max(img))


def get_foreground(img, attention):
    img = tf.math.multiply(attention, img)
    return tf.math.divide(img, tf.math.reduce_max(img))


def get_background(img, attention):
    img = tf.math.multiply(tf.math.subtract(1, attention), img)
    return tf.math.divide(img, tf.math.reduce_max(img))


class AttentionImage():
    def __init__(self, img, attention):
        self.foreground = None
        self.background = None
        self.transformed_foreground = None
        self.get_fore_and_backgrouds_by_attention(img, attention)

    def get_fore_and_backgrouds_by_attention(self, img, attention):
        # Scale all [0,1]
        img = scale_to_zero_one(img)
        attention = scale_to_zero_one(attention)
        # Split background and foreground
        self.foreground = scale_to_minus_one_one(get_foreground(img, attention))
        self.background = scale_to_minus_one_one(get_background(img, attention))
