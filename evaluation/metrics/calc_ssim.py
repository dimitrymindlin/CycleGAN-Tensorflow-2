import tensorflow as tf
from skimage.metrics import structural_similarity

from imlib import batched_tf_img_to_2d_numpy


def calc_ssim(img1, img2):
    img1 = batched_tf_img_to_2d_numpy(img1)
    img2 = batched_tf_img_to_2d_numpy(img2)
    return structural_similarity(img1, img2, multichannel=True)
