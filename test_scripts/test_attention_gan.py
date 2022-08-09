import os
import sys

from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

import pylib as py
import tensorflow as tf
import tf2lib as tl
import module

# ==============================================================================
# =                                   param                                    =
# ==============================================================================
from evaluation.kid import KID, calc_KID_for_model
from evaluation.load_test_data import load_test_data
from evaluation.tcv_os import calculate_tcv_os
from imlib import plot_any_img, scale_to_zero_one
from imlib.image_holder import ImageHolder, multiply_images, add_images

gan_model_ts = "2022-05-26--15.51"
# py.arg('--batch_size', type=int, default=32)
py.arg('--print_images', type=bool, default=True)
py.arg('--crop_size', type=int, default=256)
py.arg('--gan_model_ts', type=str, default=None)
args = py.args()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../"  # This is your Project Root
print(ROOT_DIR)
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
              py.join(f"{ROOT_DIR}/checkpoints/gans/horse2zebra/{gan_model_ts}", '../checkpoints')).restore()

save_dir = py.join(f"{ROOT_DIR}/checkpoints/gans/horse2zebra/{gan_model_ts}", 'generated_imgs')
py.mkdir(save_dir)


# restore
# tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(args.experiment_dir, 'checkpoints')).restore()
# tl.Checkpoint(dict(generator_g=G_A2B, generator_f=G_B2A), py.join(args.experiment_dir, 'checkpoints')).restore()


@tf.function
def sample_A2B(A):
    A2B = G_A2B(A, training=False)
    return A2B


@tf.function
def sample_B2A(B):
    B2A = G_B2A(B, training=False)
    return B2A


# run
clf = tf.keras.models.load_model(f"{ROOT_DIR}/checkpoints/inception_horse2zebra_512/model", compile=False)
oracle = tf.keras.models.load_model(f"{ROOT_DIR}/checkpoints/resnet50_horse2zebra_256/model", compile=False)

gradcam = GradcamPlusPlus(clf, clone=True)

# tcv, os = calculate_tcv_os(test_horses, "A2B")
# tcv, os = calculate_tcv_os(test_zebras, "B2A")

done = ["2022-05-31--14.02", "2022-05-31--13.04", "2022-06-01--13.06", "2022-06-02--12.45"]
done_ep = ["180", "180", "180", "180"]
checkpoint_ts_list = ["2022-05-31--13.04", "2022-05-31--14.02", "2022-06-01--13.06", "2022-06-02--12.45",
                      "2022-06-03--14.07", "2022-06-03--19.10"]
checkpoint_ep_list = ["180", "180", "180", "180", "160", "120"]

with open('attention_gan_run.txt', 'w') as f:
    sys.stdout = f  # Change the standard output to the file we created.
    for name, ep in zip(checkpoint_ts_list, checkpoint_ep_list):
        if name == "2022-05-23--18.32":
            tl.Checkpoint(dict(generator_g=G_A2B, generator_f=G_B2A),
                          py.join(f"{ROOT_DIR}/checkpoints/gans/horse2zebra/{name}")).restore(
                save_path=f'{ROOT_DIR}/checkpoints/gans/horse2zebra/{name}/ckpt-{ep}')
        else:
            tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A),
                          py.join(f"{ROOT_DIR}/checkpoints/gans/horse2zebra/{name}")).restore(
                save_path=f'{ROOT_DIR}/checkpoints/gans/horse2zebra/{name}/ckpt-{ep}')
        print(f"Starting {name}")
        print("-> A2B")
        save_dir = py.join(f"{ROOT_DIR}/checkpoints/gans/horse2zebra/{name}", 'generated_imgs', "A2B")
        py.mkdir(save_dir)
        tcv, os, _ = calculate_tcv_os(clf, oracle, G_A2B, G_B2A, test_horses, "A2B", gradcam)
        # calc_KID_for_model(translated_images_A2B, "A2B", args.crop_size, train_horses, train_zebras)

        print("-> B2A")
        save_dir = py.join(f"{ROOT_DIR}/checkpoints/gans/horse2zebra/{name}", 'generated_imgs', "B2A")
        py.mkdir(save_dir)
        tcv, os, _ = calculate_tcv_os(test_zebras, "B2A")
        # calc_KID_for_model(translated_images_B2A, "B2A", args.crop_size, train_horses, train_zebras)
        print("_______________________")
