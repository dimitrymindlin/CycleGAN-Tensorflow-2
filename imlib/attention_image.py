import tensorflow as tf

from imlib import scale_to_zero_one, scale_to_minus_one_one


def add_images(foreground, background):
    """
    Expects foreground and background to be in range [-1,1]
    Return image in [-1,1]
    """
    foreground = scale_to_zero_one(foreground)
    background = scale_to_zero_one(background)
    img = tf.math.add(foreground, background)
    return scale_to_minus_one_one(tf.math.divide(img, tf.math.reduce_max(img)))


def get_foreground(img, attention):
    """
    Expects img and attention to be in range [0,1]
    """
    img = tf.math.multiply(attention, img)
    return tf.math.divide(img, tf.math.reduce_max(img))


def get_background(img, attention):
    """
    Expects img and attention to be in range [0,1]
    """
    img = tf.math.multiply(tf.math.subtract(1, attention), img)
    return tf.math.divide(img, tf.math.reduce_max(img))


def multiply_images(img1, img2):
    img1 = scale_to_zero_one(img1)
    img2 = scale_to_zero_one(img2)
    img = tf.math.multiply(img1, img2)
    return scale_to_minus_one_one(tf.math.divide(img, tf.math.reduce_max(img)))


class AttentionImage():
    def __init__(self, img, attention):
        self.attention = attention
        self.foreground = None
        self.background = None
        self.transformed_part = None  # Can be either transformed foreground or transformed image with attention
        self.get_fore_and_backgrouds_by_attention(img, attention)

    def get_fore_and_backgrouds_by_attention(self, img, attention):
        # Scale all [0,1]
        img = scale_to_zero_one(img)
        attention = scale_to_zero_one(attention)
        # Split background and foreground
        self.foreground = scale_to_minus_one_one(get_foreground(img, attention))
        self.background = scale_to_minus_one_one(get_background(img, attention))
