import os
import sys

import tqdm
from tensorflow_addons.layers import InstanceNormalization
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import data

# ==============================================================================
# =                                   param                                    =
# ==============================================================================
from evaluation.kid import calc_KID_for_model

gan_model_ts = "2022-05-26--15.51"
py.arg('--dataset', default='mura')
py.arg('--datasets_dir', default='datasets')
py.arg('--batch_size', type=int, default=1)
py.arg('--print_images', type=bool, default=False)
py.arg('--crop_size', type=int, default=512)
py.arg('--gan_model_ts', type=str, default=None)

args = py.args()

# ==============================================================================
# =                                    test                                    =
# ==============================================================================
A_img_paths, B_img_paths, A_img_paths_test, B_img_paths_test = data.get_dataset_paths(args)


A_dataset, B_dataset = data.make_zip_dataset(A_img_paths, B_img_paths, args.batch_size, args.crop_size,
                                                 args.crop_size, training=False, repeat=False)

A_dataset_test, B_dataset_test = data.make_zip_dataset(A_img_paths_test, B_img_paths_test, args.batch_size, args.crop_size,
                                                 args.crop_size, training=False, repeat=False)

# run
clf = tf.keras.models.load_model(f"checkpoints/inception_{args.dataset}/2022-06-04--00.05/model", compile=False)
oracle = tf.keras.models.load_model(f"checkpoints/inception_{args.dataset}/2022-03-24--12.42/model", compile=False)
save_dir = None
gradcam = GradcamPlusPlus(clf, clone=True)


def calculate_tcv_os(dataset, translation_name, G_A2B, G_B2A):
    len_dataset = 0
    translated_images = []
    y_pred_translated = []
    y_pred_oracle = []
    for img_batch in tqdm.tqdm(dataset):
        if translation_name == "A2B":
            first_genrator = G_A2B
            cycle_generator = G_B2A

        else:
            first_genrator = G_B2A
            cycle_generator = G_A2B


        translated = first_genrator.predict(img_batch)
        normalisation_factor = np.max((np.max(translated), np.abs(np.min(translated))))
        translated /= normalisation_factor  # [-1, 1]
        # Cycle
        img_cycled = cycle_generator.predict(translated)
        normalisation_factor = np.max((np.max(img_cycled), np.abs(np.min(img_cycled))))
        img_cycled /= normalisation_factor  # [-1, 1]
        for img_i, translated_i, cycled_i in zip(img_batch, translated, img_cycled):
            translated_images.append(tf.squeeze(translated_i))
            y_pred_translated.append(
                int(np.argmax(clf(tf.expand_dims(tf.image.resize(translated_i, [512, 512]), axis=0)))))
            y_pred_oracle.append(
                int(np.argmax(oracle(tf.expand_dims(translated_i, axis=0)))))
            if args.print_images:
                img = np.concatenate([img_i, translated_i, cycled_i], axis=1)
                img_name = translation_name + "_" + str(len_dataset) + ".png"
                im.imwrite(img, py.join(save_dir, img_name))

            len_dataset += 1

    if translation_name == "A2B":
        tcv = sum(y_pred_translated) / len_dataset
        similar_predictions_count = sum(x == y == 1 for x, y in zip(y_pred_translated, y_pred_oracle))
        os = (1 / len_dataset) * similar_predictions_count
    else:
        tcv = (len_dataset - sum(y_pred_translated)) / len_dataset
        similar_predictions_count = sum(x == y == 0 for x, y in zip(y_pred_translated, y_pred_oracle))
        os = (1 / len(y_pred_translated)) * similar_predictions_count

    print(f"TCV:", float("{0:.3f}".format(np.mean(tcv))))
    print(f"OS :", float("{0:.3f}".format(np.mean(os))))
    return tcv, os, translated_images


done = [ ]
checkpoint_ts_list = ["GANterfactual_2022-03-29--00.56", "GANterfactual_2022-03-26--06.18"]

with open(f'ganterfactual_{args.dataset}.txt', 'w') as f:
    sys.stdout = f  # Change the standard output to the file we created.
    for name in checkpoint_ts_list:
        cyclegan_folder = f"checkpoints/gans/mura/{name}"
        custom_objects = {"InstanceNormalization": InstanceNormalization}
        G_A2B = tf.keras.models.load_model(os.path.join(cyclegan_folder, 'generator_np.h5'),
                                          custom_objects=custom_objects)
        G_B2A = tf.keras.models.load_model(os.path.join(cyclegan_folder, 'generator_pn.h5'),
                                          custom_objects=custom_objects)
        if name != "GANterfactual_2022-03-26--06.18":
            print(f"Starting {name}")
            print("-> A2B")
            save_dir = py.join(f"checkpoints/gans/{args.dataset}/{name}", 'generated_imgs', "A2B")
            py.mkdir(save_dir)
            _, _, translated_images_A2B = calculate_tcv_os(A_dataset_test, "A2B", G_A2B, G_B2A)
            calc_KID_for_model(translated_images_A2B, "A2B", args.crop_size, A_dataset, B_dataset)

        print("-> B2A")
        save_dir = py.join(f"checkpoints/gans/{args.dataset}/{name}", 'generated_imgs', "B2A")
        py.mkdir(save_dir)
        _, _, translated_images_B2A = calculate_tcv_os(B_dataset_test, "B2A", G_A2B, G_B2A)
        calc_KID_for_model(translated_images_B2A, "B2A", args.crop_size, A_dataset, B_dataset)
        print("_______________________")
