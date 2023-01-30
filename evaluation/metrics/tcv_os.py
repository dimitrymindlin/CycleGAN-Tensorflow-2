import os

import numpy as np
import tensorflow as tf
import tqdm
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from attention_strategies.attention_gan import attention_gan_single
from attention_strategies.no_attention import no_attention_single
from imlib import immerge, imwrite
from imlib.image_holder import ImageHolder


def only_save_imgs(args):
    """
    Returns true when all metrics are set to false, else, false
    """
    return not args.kid and not args.tcv_os and not args.ssim_psnr


def get_predicted_class_label(args, img, clf):
    if args.clf_input_channel == 1 and args.img_channels == 3:
        img = tf.image.rgb_to_grayscale(img)
    return int(np.argmax(clf(img)))


def translate_images_clf(args, dataset, clf, generator, gradcam, class_label, return_images,
                         training=False, save_img=False):
    """
    Method uses generators to translate images to new domain and uses the classifier to predict the label of the
    translated image.
    """
    use_attention = True if args.attention_type != "none" else False
    translated_images = []
    y_pred_translated = []
    len_dataset = 0

    for batch_i, img_batch in enumerate(tqdm.tqdm(dataset, desc='Translating images')):
        img_holder = ImageHolder(img_batch, args, class_label, attention_func=gradcam, use_attention=use_attention)
        class_label_name = "A" if class_label == 0 else "B"
        target_class_name = "B" if class_label == 0 else "A"
        # Generate Images (only batch of 1 img supported at the moment)
        if args.attention_type == "attention-gan-original":
            translated_img, _ = attention_gan_single(img_holder.img, generator, None, img_holder.attention,
                                                     img_holder.background, training)
        else:
            translated_img = no_attention_single(img_holder.img, generator, None, training)

        if return_images:
            # save imgs to list and return later
            """translated_img = tf.squeeze(translated_img) #unbatch 
            if args.img_channels == 1:
                translated_img = tf.expand_dims(translated_img, axis=-1)"""
            translated_images.append(translated_img)

        # Predict images with CLF
        for img_i, translated_i in zip(img_batch, translated_img):
            # To predict, scale to 512,512 and batch
            translated_i_batched = tf.expand_dims(tf.image.resize(translated_i, [512, 512]), axis=0)
            clf_prediction = get_predicted_class_label(args, translated_i_batched, clf)
            y_pred_translated.append(clf_prediction)
            # If images should be saved
            if save_img and len_dataset < 100:
                # Save first 100 imgs (original, (attention,) translated)
                if not args.save_only_translated_img:
                    original_img_batched = tf.expand_dims(tf.image.resize(img_i, [512, 512]), axis=0)
                    original_prediction = get_predicted_class_label(args, original_img_batched, clf)
                    class_label_name = "A" if class_label == 0 else "B"
                    if args.attention_type != "none":
                        img = immerge(
                            np.concatenate([img_holder.img, img_holder.attention, translated_img], axis=0), n_rows=1)
                    else:
                        img = immerge(
                            np.concatenate([img_holder.img, translated_img], axis=0), n_rows=1)
                    img_folder = f'{save_img}/{class_label_name}'
                    os.makedirs(img_folder, exist_ok=True)
                    imwrite(img, f"{img_folder}/%d_{original_prediction}_{clf_prediction}.png" % (batch_i))
                else:
                    # if only translated imgs should be saved.
                    # im = Image.fromarray(np.squeeze(np.array(0.5 * translated_img + 0.5)))
                    im = Image.fromarray(np.uint8(np.squeeze(np.array(0.5 * translated_img + 0.5)) * 255))
                    img_folder = f'{save_img}/{class_label_name}'
                    os.makedirs(img_folder, exist_ok=True)
                    im.save(f"{img_folder}/%d.png" % (batch_i))
            else:
                if only_save_imgs(args):
                    return y_pred_translated, len_dataset, translated_images  # Stop if only images should be saved.
        len_dataset += 1

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


def calculate_tcv(y_pred_translated, len_dataset, translation_name):
    # Calculate tcv
    if translation_name == "A2B":
        tcv = sum(y_pred_translated) / len_dataset
    else:
        tcv = (len_dataset - sum(y_pred_translated)) / len_dataset
    print("TCV: ", float("{0:.3f}".format(np.mean(tcv))))
    return tcv


def calculate_ssim_psnr(args, images, translated_images):
    ssim_count = 0
    psnr_count = 0
    for img_i, translated_i in zip(images, translated_images):
        if tf.shape(img_i)[-1] == 1:
            img_i = tf.image.grayscale_to_rgb(img_i)
        img_i = tf.squeeze(img_i).numpy()
        if tf.shape(translated_i)[-1] == 1:
            translated_i = tf.image.grayscale_to_rgb(translated_i)
        translated_i = tf.squeeze(translated_i).numpy()
        ssim_count += structural_similarity(img_i, translated_i, channel_axis=2)
        psnr_count += peak_signal_noise_ratio(img_i, translated_i)

    ssim = ssim_count / len(images)
    psnr = psnr_count / len(images)
    print("SSIM: ", float("{0:.3f}".format(np.mean(ssim))))
    print("PSNR: ", float("{0:.3f}".format(np.mean(psnr))))
    return ssim, psnr
