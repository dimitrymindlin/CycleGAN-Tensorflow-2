import numpy as np
import tensorflow as tf
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

import attention_maps
from imlib import scale_between_zero_one, scale_between_minus_one_one


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


class ImageHolder():
    def __init__(self, img, args, class_label=None, attention_func=None, use_attention=True,
                 attention_intensity=1, attention_source="clf", model=None, attention=None):
        self.img = img  # original image
        self.attention = None  # heatmap [-1, 1]
        self.foreground = None  # original image + heatmap
        self.background = None  # original image - heatmap
        self.enhanced_img = None  # original image * (heatmap + intensity)
        self.transformed_part = None  # depending on strategy, the part that should be transformed
        self.model = model
        if use_attention:
            if attention is None:
                self.get_attention(class_label, attention_func, args, attention_intensity, attention_source)
            else:
                self.attention = attention
                self.enhanced_img = attention_maps.apply_attention_on_img(img, attention)
            self.split_fore_and_background_by_attention()

    def get_attention(self, class_label, attention_func, args, attention_intensity, attention_source):
        # enhanced_img, attention = attention_maps.apply_lime(self.img, self.model, class_index=class_label)
        enhanced_img, attention = attention_maps.get_clf_attention_img(self.img, attention_func, class_label,
                                                                       args.attention_type,
                                                                       attention_intensity=attention_intensity,
                                                                       attention_source=attention_source)
        """else:
            enhanced_img, attention = attention_maps.apply_occlusion_sensitivity(self.img, attention_func, class_label,
                                                                                 args,
                                                                                 attention_intensity=attention_intensity,
                                                                                 attention_source=attention_source)
"""
        self.enhanced_img = enhanced_img
        self.attention = attention

    def split_fore_and_background_by_attention(self):
        # Scale all [0,1]
        img = scale_between_zero_one(self.img)
        attention = scale_between_zero_one(self.attention)
        # Split background and foreground
        self.foreground = get_foreground(img, attention)
        self.background = get_background(img, attention)


def get_img_holders(A, B, args, attention_intensity=1, attention_func=None, gradcam_D_A=None,
                    gradcam_D_B=None, model=None):
    if args.attention_type == "none":
        A_holder = ImageHolder(A, args, 0, use_attention=False)
        B_holder = ImageHolder(B, args, 1, use_attention=False)
    else:  # attention gan or spa-gan with clf attention
        B_holder = ImageHolder(B, args, 1, attention_func, attention_intensity=attention_intensity, model=model)
        A_holder = ImageHolder(A, args, 0, attention_func, attention_intensity=attention_intensity, model=model)

    """elif args.attention_type == "spa-gan": Remove spa-gan for now
        if args.attention == "discriminator":
            # Both classes here 0 because the gradcam is for each discriminator and not the classifier.
            A_holder = ImageHolder(A, 0, gradcam_D_A, args.attention_type,
                                   attention_intensity=attention_intensity, use_attention=args.attention,
                                   attention_source=args.attention)
            B_holder = ImageHolder(B, 0, gradcam_D_B, args.attention_type,
                                   attention_intensity=attention_intensity, use_attention=args.attention,
                                   attention_source=args.attention)
        else:
            A_holder = ImageHolder(A, 0, gradcam, args.attention_type, attention_intensity=attention_intensity)
            B_holder = ImageHolder(B, 1, gradcam, args.attention_type, attention_intensity=attention_intensity)"""

    return A_holder, B_holder


def get_img_holders_precomputed_attention(A, A_attention, B, B_attention, args):
    if args.attention_type == "none":
        A_holder = ImageHolder(A, args, 0, use_attention=False)
        B_holder = ImageHolder(B, args, 1, use_attention=False)
    else:  # attention gan or spa-gan with clf attention
        B_holder = ImageHolder(B, args, 1, attention=B_attention)
        A_holder = ImageHolder(A, args, 0, attention=A_attention)
    return A_holder, B_holder
