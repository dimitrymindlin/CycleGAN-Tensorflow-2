import numpy as np
import tqdm

from imlib.image_holder import ImageHolder, multiply_images, add_images
import tensorflow as tf


@tf.function
def sample(G, img):
    """

    Parameters
    ----------
    G: Generator
    img: Image

    Returns translated image
    -------

    """
    return G(img, training=False)


def calculate_tcv_os(clf, oracle, G_A2B, G_B2A, dataset, translation_name, gradcam, return_images=False):
    len_dataset = 0
    translated_images = []
    y_pred_translated = []
    y_pred_oracle = []
    for img_batch in tqdm.tqdm(dataset):
        if translation_name == "A2B":
            generator = G_A2B
            cycle_ganarator = G_B2A
            img_holder = ImageHolder(img_batch, 0, gradcam=gradcam, attention_type="attention-gan")
        else:
            generator = G_B2A
            cycle_ganarator = G_A2B
            img_holder = ImageHolder(img_batch, 1, gradcam=gradcam, attention_type="attention-gan")

        img_transformed = sample(generator, img_holder.img)
        # Combine new transformed image with attention -> Crop important part from transformed img
        img_transformed_attention = multiply_images(img_transformed, img_holder.attention)
        # Add background to new img
        translated_img = add_images(img_transformed_attention, img_holder.background)
        # Cycle
        img_cycled = sample(cycle_ganarator, translated_img)
        # Combine new transformed image with attention
        img_cycled_attention = multiply_images(img_cycled, img_holder.attention)
        cycled_img = add_images(img_cycled_attention, img_holder.background)

        for img_i, translated_i, cycled_i in zip(img_batch, translated_img, cycled_img):
            if return_images:
                translated_images.append(tf.squeeze(translated_i))
            y_pred_translated.append(
                int(np.argmax(clf(tf.expand_dims(tf.image.resize(translated_i, [512, 512]), axis=0)))))
            y_pred_oracle.append(
                int(np.argmax(oracle(tf.expand_dims(translated_i, axis=0)))))

            len_dataset += 1

    if translation_name == "A2B":
        tcv = sum(y_pred_translated) / len_dataset
        similar_predictions_count = sum(x == y == 1 for x, y in zip(y_pred_translated, y_pred_oracle))
        os = (1 / len_dataset) * similar_predictions_count
    else:
        tcv = (len_dataset - sum(y_pred_translated)) / len_dataset
        similar_predictions_count = sum(x == y == 0 for x, y in zip(y_pred_translated, y_pred_oracle))
        os = (1 / len(y_pred_translated)) * similar_predictions_count

    if return_images:
        return tcv, os, translated_images
    else:
        return tcv, os
