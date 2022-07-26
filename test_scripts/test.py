import sys

import tqdm
import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tf2lib as tl
import tensorflow_datasets as tfds
import module

# ==============================================================================
# =                                   param                                    =
# ==============================================================================
from evaluation.kid import calc_KID_for_model
from evaluation.load_test_data import load_test_data

gan_model_ts = "2022-05-26--15.51"
py.arg('--experiment_dir', default=f"checkpoints/gans/horse2zebra/{gan_model_ts}")
py.arg('--batch_size', type=int, default=32)
py.arg('--print_images', type=bool, default=False)
py.arg('--crop_size', type=int, default=256)
py.arg('--gan_model_ts', type=str, default=None)
args = py.args()
# args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
# args.__dict__.update(test_args.__dict__)

# ==============================================================================
# =                                    test                                    =
# ==============================================================================
train_horses, test_horses, train_zebras, test_zebras = load_test_data()
# model
G_A2B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))
G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))

tl.Checkpoint(dict(generator_g=G_A2B, generator_f=G_B2A),
              py.join(f"checkpoints/gans/horse2zebra/{gan_model_ts}", '../checkpoints')).restore()

save_dir = py.join(f"checkpoints/gans/horse2zebra/{gan_model_ts}", 'generated_imgs')
py.mkdir(save_dir)


# restore
# tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(args.experiment_dir, 'checkpoints')).restore()
# tl.Checkpoint(dict(generator_g=G_A2B, generator_f=G_B2A), py.join(args.experiment_dir, 'checkpoints')).restore()


@tf.function
def sample_A2B(A):
    A2B = G_A2B(A, training=False)
    A2B2A = G_B2A(A2B, training=False)
    return A2B, A2B2A


@tf.function
def sample_B2A(B):
    B2A = G_B2A(B, training=False)
    B2A2B = G_A2B(B2A, training=False)
    return B2A, B2A2B


# run
clf = tf.keras.models.load_model(f"checkpoints/inception_horse2zebra_512/model", compile=False)
oracle = tf.keras.models.load_model(f"checkpoints/resnet50_horse2zebra_256/model", compile=False)


def calculate_tcv_os(dataset, translation_name):
    len_dataset = 0
    translated_images = []
    y_pred_translated = []
    y_pred_oracle = []
    for img_batch in tqdm.tqdm(dataset):
        if translation_name == "A2B":
            translated_img_batch, cycled_img_batch = sample_A2B(img_batch)
        else:
            translated_img_batch, cycled_img_batch = sample_B2A(img_batch)
        for img_i, translated_i, cycled_i in zip(img_batch, translated_img_batch, cycled_img_batch):
            translated_images.append(tf.squeeze(translated_i))
            # img_i = img_i.numpy()
            # translated_i = translated_i.numpy()
            y_pred_translated.append(
                int(np.argmax(clf(tf.expand_dims(tf.image.resize(translated_i, [512, 512]), axis=0)))))
            y_pred_oracle.append(
                int(np.argmax(oracle(tf.expand_dims(translated_i, axis=0)))))
            if args.print_images:
                """img = immerge(np.concatenate([img_i.numpy(), translated_i.numpy(), cycled_i.numpy()], axis=0), n_rows=1)
                imwrite(img, py.join(save_dir, translation_name + "_" + str(len_dataset)))"""

                img = np.concatenate([img_i.numpy(), translated_i.numpy(), cycled_i.numpy()], axis=1)
                img_name = translation_name + "_" + str(len_dataset) + ".png"
                im.imwrite(img, py.join(save_dir, img_name))

            # ssim_count += structural_similarity(img_i, translated_i, channel_axis=2, data_range=2)
            # psnr_count += peak_signal_noise_ratio(img_i, translated_i, data_range=2)
            len_dataset += 1

    if translation_name == "A2B":
        tcv = sum(y_pred_translated) / len_dataset
        similar_predictions_count = sum(x == y == 1 for x, y in zip(y_pred_translated, y_pred_oracle))
        os = (1 / len_dataset) * similar_predictions_count
    else:
        tcv = (len_dataset - sum(y_pred_translated)) / len_dataset
        similar_predictions_count = sum(x == y == 0 for x, y in zip(y_pred_translated, y_pred_oracle))
        os = (1 / len(y_pred_translated)) * similar_predictions_count

    # print(f"Results for {translation_name}")
    # print(f"SSIM: ", ssim_count / len_dataset)
    # print(f"PSNR: ", psnr_count / len_dataset)
    print(f"TCV:", float("{0:.3f}".format(np.mean(tcv))))
    print(f"OS :", float("{0:.3f}".format(np.mean(os))))
    return tcv, os, translated_images


# tcv, os = calculate_tcv_os(test_horses, "A2B")
# tcv, os = calculate_tcv_os(test_zebras, "B2A")

done = ["2022-05-31--14.17", "2022-05-23--18.32", "2022-05-26--08.38", "2022-06-02--12.30", ]
done_ep = ["180", "39", "140", "180"]
checkpoint_ts_list = ["2022-05-26--08.38", "2022-06-02--12.30", ]
checkpoint_ep_list = [ "140", "180"]

with open('normal_run.txt', 'w') as f:
    sys.stdout = f  # Change the standard output to the file we created.
    for name, ep in zip(checkpoint_ts_list, checkpoint_ep_list):
        if name == "2022-05-23--18.32":
            tl.Checkpoint(dict(generator_g=G_A2B, generator_f=G_B2A),
                          py.join(f"checkpoints/gans/horse2zebra/{name}")).restore(
                save_path=f'checkpoints/gans/horse2zebra/{name}/ckpt-{ep}')
        else:
            tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A),
                          py.join(f"checkpoints/gans/horse2zebra/{name}")).restore(
                save_path=f'checkpoints/gans/horse2zebra/{name}/ckpt-{ep}')
        print(f"Starting {name}")
        print("-> A2B")
        save_dir = py.join(f"checkpoints/gans/horse2zebra/{name}", 'generated_imgs', "A2B")
        py.mkdir(save_dir)
        _, _, translated_images_A2B = calculate_tcv_os(test_horses, "A2B")
        calc_KID_for_model(translated_images_A2B, "A2B", args.crop_size, train_horses, train_zebras)

        """print("-> B2A")
        save_dir = py.join(f"checkpoints/gans/horse2zebra/{name}", 'generated_imgs', "B2A")
        py.mkdir(save_dir)
        _, _, translated_images_B2A = calculate_tcv_os(test_zebras, "B2A")
        calc_KID_for_model(translated_images_B2A, "B2A", args.crop_size, train_horses, train_zebras)
        print("_______________________")"""
