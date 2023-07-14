import numpy as np
import tensorflow as tf

from imlib import scale_between_zero_one, scale_between_minus_one_one, attention_maps


def add_images(foreground, background):
    """
    Expects foreground and background to be in range [-1,1]
    Return image in [-1,1]
    """
    foreground = scale_between_zero_one(foreground)
    background = scale_between_zero_one(background)
    new = tf.math.add(foreground, background)
    return scale_between_minus_one_one(tf.math.divide(new, tf.math.reduce_max(new)))


def get_foreground(img, attention):
    """
    Expects img and attention to be in range [0,1]
    """
    new = tf.math.multiply(attention, img)
    if np.min(new) == 0 and np.max(new) == 0:
        return new
    return scale_between_minus_one_one(tf.math.divide(new, tf.math.reduce_max(new)))


def get_background(img, attention):
    """
    Expects img and attention to be in range [0,1]
    """
    new = tf.math.multiply(tf.math.subtract(1, attention), img)
    if np.min(new) == 0 and np.max(new) == 0:
        return new
    return scale_between_minus_one_one(tf.math.divide(new, tf.math.reduce_max(new)))


def multiply_images(img1, img2):
    img1 = scale_between_zero_one(img1)
    img2 = scale_between_zero_one(img2)
    new = tf.math.multiply(img1, img2)
    return scale_between_minus_one_one(new)


class ImageSegmentation:
    def __init__(self, img, args, class_label=None, attention_func=None, use_attention=True, model=None,
                 attention=None):
        self.img = img  # original image
        self.attention = None  # attention map in range [-1, 1]
        self.foreground = None  # original image + attention map
        self.background = None  # original image - attention map
        self.enhanced_img = None  # original image * (attention map  + intensity)
        self.transformed_part = None  # depending on strategy, the part that should be transformed
        self.model = model
        if use_attention:
            if attention is None:
                self.get_attention(class_label, attention_func, args)
            else:
                self.attention = attention
                self.enhanced_img = attention_maps.apply_attention_on_img(img, attention)
            self.split_fore_and_background_by_attention()

    def get_attention(self, class_label, attention_func, args):
        enhanced_img, attention = attention_maps.get_clf_attention_img(self.img, attention_func, class_label, args)
        self.enhanced_img = enhanced_img
        self.attention = attention

    def split_fore_and_background_by_attention(self):
        # Scale all [0,1]
        img = scale_between_zero_one(self.img)
        attention = scale_between_zero_one(self.attention)
        # Split background and foreground
        self.foreground = get_foreground(img, attention)
        self.background = get_background(img, attention)


def get_img_segmentations(A, B, args, attention_func, model):
    A_img_seg = ImageSegmentation(A, args, 0, attention_func, model=model)
    B_img_seg = ImageSegmentation(B, args, 1, attention_func, model=model)
    return A_img_seg, B_img_seg
