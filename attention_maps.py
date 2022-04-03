import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from tf_keras_vis.utils.scores import CategoricalScore

"""AUGMENTATIONS_TEST = Compose([
    CLAHE(always_apply=True)
])


def find_target_layer(model):
    # attempt to find the final convolutional layer in the network
    # by looping over the layers of the network in reverse order
    for idx, layer in enumerate(reversed(model.layers)):
        # check to see if the layer has a 4D output
        try:
            if len(layer.output_shape) == 4:
                return layer.name, idx
        except AttributeError:
            print("Output ...")
    # otherwise, we could not find a 4D layer so the GradCAM
    # algorithm cannot be applied
    raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")


def make_heatmap(img_array, model):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    last_conv_layer_name, _ = find_target_layer(model)

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
        tape.watch(class_channel)

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    # Rescale heatmap to a range 0-255
    #heatmap = np.uint8(255 * heatmap)
    heatmap = tf.math.multiply(255, tf.cast(heatmap, tf.int32))

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]

    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[2]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = tf.math.subtract(tf.math.divide(jet_heatmap * 0.005 + img_array, 127.5), 1)  # scale [-1,1]

    return superimposed_img, jet_heatmap


def generate_gradcam_attention(model, img):
    return make_heatmap(img, model)"""


def get_attention_image(img, gradcam):
    # Generate cam with GradCAM++ for A
    cam = gradcam(CategoricalScore(0), img)  # [0,1]
    cam = tf.expand_dims(cam, axis=-1)
    cam = tf.image.grayscale_to_rgb(tf.convert_to_tensor(cam)) # [1,512,512,3] like img

    # Rescale img to 0,1
    img = 0.5 * img + 0.5  # [0, 1]

    # Interpolate by addition and normalise back to 0,1
    img = tf.math.add(img, cam)
    img /= np.max(img)

    return img * 2.0 - 1, cam * 2.0 -1  # [-1,1]
