import os
import sys

from mura.tfds_from_disc import get_mura_test_ds_by_body_part_split_class

from pylib import load_args
from test_config import config
from rsna import get_rsna_TEST_ds_split_class
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
import pylib as py
import tensorflow as tf
import tf2lib_local as tl
import module
from evaluation.kid import calc_KID_for_model_batched, calc_KID_for_model_target_source_batched
from evaluation.load_test_data import load_tfds_test_data
from evaluation.tcv_os import calculate_ssim_psnr, calculate_tcv, translate_images_clf
from tensorflow_addons.layers import InstanceNormalization

from tf2lib_local.utils import is_ganterfactual_repo

# ==============================================================================
# =                                   param                                    =
# ==============================================================================
py.arg('--dataset', default='rsna', choices=['horse2zebra', 'mura', 'apple2orange', 'rsna'])
py.arg('--save_img', type=bool, default=True)
py.arg('--save_only_translated_img', type=bool, default=False)
py.arg('--tcv_os', type=bool, default=True)
py.arg('--ssim_psnr', type=bool, default=True)
py.arg('--kid', type=bool, default=True)
py.arg('--generator', type=str, default="resnet", choices=['resnet', 'unet'])

args = py.args()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
experiments_dir = f"{ROOT_DIR}/checkpoints/gans/{args.dataset}/"  # CycleGAN experiment results folder
TFDS_PATH = f"{ROOT_DIR}/../tensorflow_datasets"  # Path to datasets
KID = args.kid
TCV = args.tcv_os
SSIM = args.ssim_psnr
SAVE_IMG = args.save_img
SAVE_ONLY_TRANSLATED_IMG = args.save_only_translated_img
print(TFDS_PATH)


def get_abc_gan_generators(timestamp_id, ep):
    if timestamp_id == "2022-05-23--18.32":  # normal MURA gan without attention
        tl.utils.Checkpoint(dict(generator_g=G_A2B, generator_f=G_B2A),
                            py.join(f"{ROOT_DIR}/checkpoints/gans/{args.dataset}/{timestamp_id}")).restore(
            save_path=f'{ROOT_DIR}/checkpoints/gans/{args.dataset}/{timestamp_id}/ckpt-{ep}')
    else:
        if os.path.exists(py.join(f"{ROOT_DIR}/checkpoints/gans/{args.dataset}/{timestamp_id}/checkpoints")):
            # New runs will keep checkpoints in checkpoints folder
            tl.utils.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A),
                                py.join(f"{ROOT_DIR}/checkpoints/gans/{args.dataset}/{timestamp_id}/checkpoints")).restore(
                save_path=f'{ROOT_DIR}/checkpoints/gans/{args.dataset}/{timestamp_id}/checkpoints/ckpt-{ep}')
        else:
            tl.utils.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A),
                                py.join(f"{ROOT_DIR}/checkpoints/gans/{args.dataset}/{timestamp_id}")).restore(
                save_path=f'{ROOT_DIR}/checkpoints/gans/{args.dataset}/{timestamp_id}/ckpt-{ep}')
    return G_A2B, G_B2A


def get_ganterfactual_generators(name, ep):
    cyclegan_folder = f"{ROOT_DIR}/checkpoints/gans/{args.dataset}/{name}/{ep}"
    custom_objects = {"InstanceNormalization": InstanceNormalization}
    G_A2B = tf.keras.models.load_model(os.path.join(cyclegan_folder, 'generator_np.h5'),
                                       custom_objects=custom_objects)
    G_B2A = tf.keras.models.load_model(os.path.join(cyclegan_folder, 'generator_pn.h5'),
                                       custom_objects=custom_objects)
    return G_A2B, G_B2A


# ==============================================================================
# =                                    data                                    =
# ==============================================================================
def load_test_data():
    if args.dataset == "mura":
        A_dataset, B_dataset, A_dataset_test, B_dataset_test = get_mura_test_ds_by_body_part_split_class(
            args.body_parts,
            TFDS_PATH,
            args.batch_size,
            args.crop_size,
            args.crop_size,
            special_normalisation=None)
    elif args.dataset == "rsna":
        A_dataset, B_dataset, A_dataset_test, B_dataset_test = get_rsna_TEST_ds_split_class(TFDS_PATH,
                                                                                            args.batch_size,
                                                                                            args.crop_size,
                                                                                            args.crop_size,
                                                                                            special_normalisation=None,
                                                                                            channels=args.img_channels,
                                                                                            training=False)


    else:  # Horse2Zebra / Apple2Orange
        A_dataset, A_dataset_test, B_dataset, B_dataset_test = load_tfds_test_data(args.dataset)
    """print("A_dataset",len(A_dataset))
    print("A_dataset_test", len(A_dataset_test))
    print("B_dataset", len(B_dataset))
    print("B_dataset_test", len(B_dataset_test))"""
    return A_dataset, A_dataset_test, B_dataset, B_dataset_test


# ==============================================================================
# =                                    models                                  =
# ==============================================================================
def get_load_generators():
    """
    Decide if generators from this project or GANterfactual project.
    """
    if is_ganterfactual_repo(args):
        load_generators = get_ganterfactual_generators
    else:
        load_generators = get_abc_gan_generators
    return load_generators


def evaluate_current_model(G_A2B, G_B2A, A_dataset, A_dataset_test, B_dataset, B_dataset_test, save_img=False):
    for translation_name in ["B2A", "A2B"]:
        print(f"-> {translation_name}")
        if translation_name == "A2B":
            generator = G_A2B
            class_label = 0
            source_dataset = A_dataset_test
            target_dataset = B_dataset
        else:
            generator = G_B2A
            class_label = 1
            source_dataset = B_dataset_test
            target_dataset = A_dataset

        # Get counterfactuals (translated images)
        y_pred_translated, len_dataset, translated_images = translate_images_clf(args,
                                                                                 source_dataset, clf, generator,
                                                                                 gradcam, class_label, True,
                                                                                 training=False, save_img=save_img)

        if args.tcv_os:
            calculate_tcv(y_pred_translated, len_dataset, translation_name)

        if args.ssim_psnr:
            calculate_ssim_psnr(args, source_dataset, translated_images)

        if args.kid:
            if args.dataset == "mura" or args.dataset == "rsna":
                calc_KID_for_model_batched(translated_images, args.img_shape, target_dataset)
            else:
                calc_KID_for_model_target_source_batched(translated_images, args.img_shape, A_dataset,
                                                         B_dataset, translation_name)
    print()


for name, ep in zip(config[args.dataset]["model_names"], config[args.dataset]["epochs"]):
    args = load_args(args, py.join(experiments_dir, name))
    args.kid = KID
    args.tcv_os = TCV
    args.ssim_psnr = SSIM
    args.save_img = SAVE_IMG
    args.save_only_translated_img = SAVE_ONLY_TRANSLATED_IMG

    ### Get Generators
    load_generators = get_load_generators()  # decide which load method to use
    generator = module.ResnetGenerator if args.generator == "resnet" else module.UnetGenerator
    G_A2B = generator(input_shape=args.img_shape)
    G_B2A = generator(input_shape=args.img_shape)

    G_A2B, G_B2A = load_generators(name, ep)  # load generator checkpoints.
    ### Get CLF + Gradcam
    clf = tf.keras.models.load_model(
        f"{ROOT_DIR}/checkpoints/{args.clf_name}_{args.dataset}/{args.clf_ckp_name}/model", compile=False)
    if args.attention_type != "none":
        gradcam = GradcamPlusPlus(clf, clone=True)
    else:
        gradcam = None
    # Set clf input shape if not done:
    try:
        args.clf_input_channel
    except AttributeError:
        args.clf_input_channel = clf.layers[0].input_shape[0][-1]  # ignore batch dimension eg. (512,512,3)

    with open(py.join(experiments_dir, name, f'test_metrics_{ep}.txt'), 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(f"Starting {name}_{ep}")
        if args.save_img:
            if args.dataset == "apple2orange":
                save_img = py.join(experiments_dir, name, f"test_images_target_source_{ep}")
            else:
                save_img = py.join(experiments_dir, name, f"test_images_{ep}")
        else:
            save_img = False
        A_dataset, A_dataset_test, B_dataset, B_dataset_test = load_test_data()
        evaluate_current_model(G_A2B, G_B2A, A_dataset, A_dataset_test, B_dataset, B_dataset_test, save_img)
