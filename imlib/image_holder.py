import tensorflow as tf

import attention_maps
from imlib import scale_to_zero_one, scale_to_minus_one_one
import numpy as np


def add_images(foreground, background):
    """
    Expects foreground and background to be in range [-1,1]
    Return image in [-1,1]
    """
    foreground = scale_to_zero_one(foreground)
    background = scale_to_zero_one(background)
    new = tf.math.add(foreground, background)
    return scale_to_minus_one_one(tf.math.divide(new, tf.math.reduce_max(new)))


def get_foreground(img, attention):
    """
    Expects img and attention to be in range [0,1]
    """
    new = tf.math.multiply(attention, img)
    if np.min(new) == 0 and np.max(new) == 0:
        return new
    return scale_to_minus_one_one(tf.math.divide(new, tf.math.reduce_max(new)))


def get_background(img, attention):
    """
    Expects img and attention to be in range [0,1]
    """
    new = tf.math.multiply(tf.math.subtract(1, attention), img)
    if np.min(new) == 0 and np.max(new) == 0:
        return new
    return scale_to_minus_one_one(tf.math.divide(new, tf.math.reduce_max(new)))


def multiply_images(img1, img2):
    img1 = scale_to_zero_one(img1)
    img2 = scale_to_zero_one(img2)
    new = tf.math.multiply(img1, img2)
    return scale_to_minus_one_one(new)


class ImageHolder():
    def __init__(self, img, class_label=None, gradcam=None, attention_type=None, use_attention=True,
                 attention_intensity=1, attention_source="clf"):
        self.img = img  # original image
        self.attention = None  # heatmap
        self.foreground = None  # original image + heatmap
        self.background = None  # original image - heatmap
        self.enhanced_img = None  # original image * (heatmap + intensity)
        self.transformed_part = None  # depending on strategy, the part that should be transformed
        if use_attention:
            self.get_attention(class_label, gradcam, attention_type, attention_intensity, attention_source)
            self.split_fore_and_background_by_attention()

    def get_attention(self, class_label, gradcam, attention_type, attention_intensity, attention_source):
        enhanced_img, attention = attention_maps.apply_gradcam(self.img, gradcam, class_label,
                                                               attention_type=attention_type,
                                                               attention_intensity=attention_intensity,
                                                               attention_source=attention_source)

        self.enhanced_img = enhanced_img
        self.attention = attention

    def split_fore_and_background_by_attention(self):
        # Scale all [0,1]
        img = scale_to_zero_one(self.img)
        attention = scale_to_zero_one(self.attention)
        # Split background and foreground
        self.foreground = get_foreground(img, attention)
        self.background = get_background(img, attention)


def get_img_holders(A, B, attention_type, attention, attention_intensity=None, gradcam=None, gradcam_D_A=None,
                    gradcam_D_B=None):
    if attention_type == "none":
        A_holder = ImageHolder(A, 0, use_attention=False, attention_intensity=attention_intensity)
        B_holder = ImageHolder(B, 1, use_attention=False, attention_intensity=attention_intensity)
    elif attention_type == "spa-gan":
        if attention == "discriminator":
            # Both classes here 0 because the gradcam is for each discriminator and not the classifier.
            A_holder = ImageHolder(A, 0, gradcam_D_A, attention_type,
                                   attention_intensity=attention_intensity, use_attention=attention)
            B_holder = ImageHolder(B, 0, gradcam_D_B, attention_type,
                                   attention_intensity=attention_intensity, use_attention=attention)
        else:
            A_holder = ImageHolder(A, 0, gradcam, attention_type, attention_intensity=attention_intensity)
            B_holder = ImageHolder(B, 1, gradcam, attention_type, attention_intensity=attention_intensity)
    else:  # attention gan or spa-gan with clf attention
        A_holder = ImageHolder(A, 0, gradcam, attention_type, attention_intensity=attention_intensity)
        B_holder = ImageHolder(B, 1, gradcam, attention_type, attention_intensity=attention_intensity)

    return A_holder, B_holder
