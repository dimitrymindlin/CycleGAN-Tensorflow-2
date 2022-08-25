import os

import numpy as np
from matplotlib import pyplot as plt

import pylib as py
import tqdm

from attention_strategies.attention_gan import attention_gan_single
from attention_strategies.no_attention import no_attention_single
from imlib import immerge, imwrite
from imlib.image_holder import ImageHolder
import tensorflow as tf


def translate_images_clf_oracle(dataset, clf, oracle, generator, gradcam, class_label, return_images, attention_type,
                                training=False, save_img=False):
    translated_images = []
    y_pred_translated = []
    y_pred_oracle = []
    len_dataset = 0

    for batch_i, img_batch in enumerate(tqdm.tqdm(dataset, desc='Translating images')):
        img_holder = ImageHolder(img_batch, class_label, gradcam=gradcam, attention_type=attention_type)
        class_label_name = "Normal" if class_label == 0 else "Abnormal"
        target_class_name = "Abnormal" if class_label == 0 else "Normal"
        if attention_type == "attention-gan":
            translated_img, _ = attention_gan_single(img_holder.img, generator, None, img_holder.attention,
                                                     img_holder.background, training)
        else:
            translated_img = no_attention_single(img_holder.img, generator, None, training)
        #
        for img_i, translated_i in zip(img_batch, translated_img):
            if return_images:
                translated_images.append(tf.squeeze(translated_i))
            clf_prediction = int(np.argmax(clf(tf.expand_dims(tf.image.resize(translated_i, [512, 512]), axis=0))))
            y_pred_translated.append(clf_prediction)
            oracle_prediction = int(np.argmax(oracle(tf.expand_dims(translated_i, axis=0))))
            y_pred_oracle.append(oracle_prediction)
            if save_img:
                """img = immerge(np.concatenate([img_holder.img, img_holder.attention, translated_img], axis=0), n_rows=1)
                class_label_name = "Normal" if class_label == 0 else "Abnormal"
                img_folder = f'output_mura/{class_label_name}'
                py.mkdir(img_folder)
                imwrite(img, f"{img_folder}/%d.png" % (batch_i))"""
                r, c = 1, 3
                titles = ['Original', 'Attention', 'Output']
                imgs = [img_holder.img, img_holder.attention, translated_img]
                original_prediction = int(np.argmax(clf(tf.expand_dims(tf.image.resize(img_i, [512, 512]), axis=0))))
                classification = [original_prediction, "", clf_prediction]
                gen_imgs = np.concatenate(imgs)
                gen_imgs = 0.5 * gen_imgs + 0.5
                correct_classification = [class_label_name, class_label_name, target_class_name]
                fig, axs = plt.subplots(r, c, figsize=(30, 20))
                cnt = 0

                for j in range(c):
                    axs[j].imshow(gen_imgs[cnt][:, :, 0], cmap='gray')
                    if j == 2:
                        axs[j].set_title(
                            f'{titles[j]} (T: {correct_classification[cnt]} | P: {classification[cnt]}, O:{oracle_prediction})')
                    elif j == 1:
                        axs[j].set_title(
                            f'{titles[j]}')
                    else:
                        axs[j].set_title(
                            f'{titles[j]} (T: {correct_classification[cnt]} | P: {classification[cnt]}')
                    axs[j].axis('off')
                    cnt += 1
                img_folder = f'{save_img}/{class_label_name}'
                os.makedirs(img_folder, exist_ok=True)
                fig.savefig(f"{img_folder}/%d.png" % (batch_i))
                plt.close()
        len_dataset += 1
    return y_pred_translated, y_pred_oracle, len_dataset, translated_images


def calculate_tcv_os(clf, oracle, G_A2B, G_B2A, dataset, translation_name, gradcam, attention_type,
                     return_images=False, save_img=False):
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
                                                                                                   return_images,
                                                                                                   save_img=save_img)
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
