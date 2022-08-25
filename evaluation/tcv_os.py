import numpy as np
import tqdm

from attention_strategies.attention_gan import attention_gan_single
from attention_strategies.no_attention import no_attention_single
from imlib.image_holder import ImageHolder
import tensorflow as tf


def translate_images_clf_oracle(dataset, clf, oracle, generator, gradcam, class_label, return_images, attention_type,
                                training=False):
    translated_images = []
    y_pred_translated = []
    y_pred_oracle = []
    len_dataset = 0

    for img_batch in tqdm.tqdm(dataset, desc='Translating images'):
        img_holder = ImageHolder(img_batch, class_label, gradcam=gradcam, attention_type=attention_type)
        if attention_type == "attention-gan":
            translated_img, _ = attention_gan_single(img_holder.img, generator, None, img_holder.attention,
                                                     img_holder.background, training)
        else:
            translated_img = no_attention_single(img_holder.img, generator, None, training)
        #
        for img_i, translated_i in zip(img_batch, translated_img):
            if return_images:
                translated_images.append(tf.squeeze(translated_i))
            y_pred_translated.append(
                int(np.argmax(clf(tf.expand_dims(tf.image.resize(translated_i, [512, 512]), axis=0)))))
            y_pred_oracle.append(
                int(np.argmax(oracle(tf.expand_dims(translated_i, axis=0)))))

        len_dataset += 1
    return y_pred_translated, y_pred_oracle, len_dataset, translated_images


def calculate_tcv_os(clf, oracle, G_A2B, G_B2A, dataset, translation_name, gradcam, attention_type,
                     return_images=False):
    if translation_name == "A2B":
        generator = G_A2B
        class_label = 0
    else:
        generator = G_B2A
        class_label = 1

    # Get translated images
    y_pred_translated, y_pred_oracle, len_dataset, translated_images = translate_images_clf_oracle(dataset, clf, oracle,
                                                                                                   generator,
                                                                                                   gradcam, class_label,
                                                                                                   attention_type,
                                                                                                   return_images)
    # Calculate tcv and os
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
