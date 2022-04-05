import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tf_keras_vis.utils.scores import CategoricalScore



def get_gradcam(img, gradcam_plus, clf=None, class_index=None):
    # Generate cam with GradCAM++ for A

    cam = gradcam_plus(CategoricalScore(class_index), img)  # [0,1]
    cam += 0.5  # ensure all values are some threshold
    cam /= np.max(cam)
    cam = tf.expand_dims(cam, axis=-1)
    cam = tf.image.grayscale_to_rgb(tf.convert_to_tensor(cam))  # [1,512,512,3] like img

    # Rescale img to 0,1
    img = 0.5 * img + 0.5  # [0, 1]

    # Interpolate by addition and normalise back to 0,1
    img = tf.math.multiply(img, cam)
    img /= np.max(img)

    """plt.imshow(np.squeeze(cam))
    plt.show()
    plt.imshow(np.squeeze(img))
    plt.show()"""

    return img * 2.0 - 1, cam * 2.0 - 1  # [-1,1]

