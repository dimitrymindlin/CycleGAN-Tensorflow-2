from skimage.color import rgb2gray
from skimage.util import compare_images

from imlib import scale_between_minus_one_one


def get_difference_img(img1, img2):
    """
    Returns difference of two images as new image as tensor.
    """

    # Convert the tensors to NumPy arrays
    image1 = img1.numpy()
    image2 = img2.numpy()

    # Convert the images to grayscale
    image1_gray = rgb2gray(image1)
    image2_gray = rgb2gray(image2)

    # Compute difference image
    difference = compare_images(image1_gray, image2_gray, method='diff')

    # Scale it to [-1,1] tensor (to fit all other images)
    difference = scale_between_minus_one_one(difference)
    return difference
