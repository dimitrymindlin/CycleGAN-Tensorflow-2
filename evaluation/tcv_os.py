import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
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
        if attention_type == "attention-gan-original":
            translated_img, _ = attention_gan_single(img_holder.img, generator, None, img_holder.attention,
                                                     img_holder.background, training)
        else:
            translated_img = no_attention_single(img_holder.img, generator, None, training)
        # Predict images with CLF and Oracle
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
        if len_dataset >= 250:
            break
    return y_pred_translated, y_pred_oracle, len_dataset, translated_images


def translate_images_clf(args, dataset, clf, generator, gradcam, class_label, return_images,
                         training=False, save_img=False):
    translated_images = []
    y_pred_translated = []
    len_dataset = 0

    for batch_i, img_batch in enumerate(tqdm.tqdm(dataset, desc='Translating images')):
        img_holder = ImageHolder(img_batch, args, class_label, gradcam=gradcam)
        class_label_name = "Normal" if class_label == 0 else "Abnormal"
        target_class_name = "Abnormal" if class_label == 0 else "Normal"
        if args.attention_type == "attention-gan-original":
            translated_img, _ = attention_gan_single(img_holder.img, generator, None, img_holder.attention,
                                                     img_holder.background, training)
        else:
            translated_img = no_attention_single(img_holder.img, generator, None, training)
        # Predict images with CLF
        for img_i, translated_i in zip(img_batch, translated_img):
            if return_images:
                translated_i = tf.squeeze(translated_i)
                if args.img_channels == 1:
                    translated_i = tf.expand_dims(translated_i, axis=-1)
                translated_images.append(translated_i)
            translated_i_batched = tf.expand_dims(tf.image.resize(translated_i, [512, 512]), axis=0)
            if args.clf_input_channel == 1:
                translated_i_batched = tf.image.rgb_to_grayscale(translated_i_batched)
            clf_prediction = int(np.argmax(clf(translated_i_batched)))
            y_pred_translated.append(clf_prediction)
            if not args.save_only_translated_img and save_img:
                original_img_batched = tf.expand_dims(tf.image.resize(img_i, [512, 512]), axis=0)
                if args.clf_input_channel == 1:
                    original_img_batched = tf.image.rgb_to_grayscale(original_img_batched)
                original_prediction = int(np.argmax(clf(original_img_batched)))
                if args.dataset == "apple2orange":
                    if args.attention != "none":
                        img = immerge(np.concatenate([img_holder.img, img_holder.attention, translated_img], axis=0),
                                      n_rows=1)
                    else:
                        img = immerge(np.concatenate([img_holder.img, translated_img], axis=0), n_rows=1)
                    class_label_name = "Normal" if class_label == 0 else "Abnormal"
                    img_folder = f'{save_img}/{class_label_name}'
                    os.makedirs(img_folder, exist_ok=True)
                    imwrite(img, f"{img_folder}/%d_{original_prediction}_{clf_prediction}.png" % (batch_i))
                else:
                    r, c = 1, 3
                    titles = ['Original', 'Attention', 'Output']
                    imgs = [img_holder.img, img_holder.attention, translated_img]

                    classification = [original_prediction, "", clf_prediction]
                    gen_imgs = np.concatenate(imgs)
                    gen_imgs = 0.5 * gen_imgs + 0.5
                    correct_classification = [class_label_name, class_label_name, target_class_name]
                    fig, axs = plt.subplots(r, c, figsize=(30, 20))
                    cnt = 0

                    cmap = 'gray' if args.dataset in ["mura", "rsna"] else None
                    for j in range(c):
                        if cmap:
                            axs[j].imshow(gen_imgs[cnt][:, :, 0], cmap=cmap)
                        else:
                            axs[j].imshow(gen_imgs[cnt][:, :, 0])
                        if j == 2:
                            axs[j].set_title(
                                f'{titles[j]} (T: {correct_classification[cnt]} | P: {classification[cnt]})')
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

            if args.save_only_translated_img:
                # im = Image.fromarray(np.squeeze(np.array(0.5 * translated_img + 0.5)))
                im = Image.fromarray(np.uint8(np.squeeze(np.array(0.5 * translated_img + 0.5)) * 255))
                img_folder = f'{save_img}/{class_label_name}'
                os.makedirs(img_folder, exist_ok=True)
                im.save(f"{img_folder}/%d.png" % (batch_i))
        len_dataset += 1
        if len_dataset > 250:
            break
    return y_pred_translated, len_dataset, translated_images


def calculate_tcv_os(y_pred_translated, y_pred_oracle, len_dataset, translation_name):
    # Calculate tcv and os
    if translation_name == "A2B":
        tcv = sum(y_pred_translated) / len_dataset
        similar_predictions_count = sum(x == y == 1 for x, y in zip(y_pred_translated, y_pred_oracle))

        os = (1 / len_dataset) * similar_predictions_count
    else:
        tcv = (len_dataset - sum(y_pred_translated)) / len_dataset
        similar_predictions_count = sum(x == y == 0 for x, y in zip(y_pred_translated, y_pred_oracle))

        os = (1 / len(y_pred_translated)) * similar_predictions_count
    print("TCV: ", tcv)
    print("OS: ", os)
    return tcv, os


def calculate_tcv(y_pred_translated, len_dataset, translation_name, calc_os=False):
    # Calculate tcv and os
    if translation_name == "A2B":
        tcv = sum(y_pred_translated) / len_dataset
    else:
        tcv = (len_dataset - sum(y_pred_translated)) / len_dataset
    print("TCV: ", float("{0:.3f}".format(np.mean(tcv))))
    return tcv


def calculate_ssim_psnr(images, translated_images):
    ssim_count = 0
    psnr_count = 0
    for img_i, translated_i in zip(images, translated_images):
        img_i = tf.squeeze(img_i)
        if np.shape(translated_images)[-1] == 1:
            img_i = tf.expand_dims(img_i, axis=-1).numpy()
        else:
            img_i = img_i.numpy()
        translated_i = translated_i.numpy()
        ssim_count += structural_similarity(img_i, translated_i, channel_axis=2)
        psnr_count += peak_signal_noise_ratio(img_i, translated_i)

    ssim = ssim_count / len(images)
    psnr = psnr_count / len(images)
    print("SSIM: ", float("{0:.3f}".format(np.mean(ssim))))
    print("PSNR: ", float("{0:.3f}".format(np.mean(psnr))))
    return ssim, psnr
