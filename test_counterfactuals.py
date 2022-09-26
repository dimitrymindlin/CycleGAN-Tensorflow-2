import os
import sys

import tqdm
from mura.tfds_from_disc import get_mura_test_ds_by_body_part_split_class
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
import pylib as py
import tensorflow as tf
import tf2lib as tl
import module
from evaluation.kid import calc_KID_for_model_target_source, calc_KID_for_model
from evaluation.load_test_data import load_tfds_test_data
from evaluation.tcv_os import calculate_ssim_psnr, calculate_tcv, translate_images_clf
from tensorflow_addons.layers import InstanceNormalization

# ==============================================================================
# =                                   param                                    =
# ==============================================================================
py.arg('--dataset', default='mura', choices=['horse2zebra', 'mura', 'apple2orange'])
py.arg('--body_parts', default=["XR_WRIST"])  # Only used in Mura dataset. Body part of x-ray images
py.arg('--batch_size', type=int, default=1)
py.arg('--datasets_dir', default='datasets')
py.arg('--attention_type', type=str, default="attention-gan-original",
       choices=['attention-gan-foreground', 'none', 'attention-gan-original'])
py.arg('--clf_name', type=str, default="inception")
py.arg('--clf_ckp_name', type=str, default="2022-06-04--00.00")  # Mura: 2022-06-04--00.05, H2Z: 2022-06-04--00.00 # A2O: 2022-09-23--15.18
py.arg('--oracle_name', type=str, default="resnet50")  # Mura: inception H2Z: resnet50
py.arg('--oracle_ckp_name', type=str, default="2022-08-21--00.00")  # Mura: 2022-03-24--12.42 H2Z: 2022-08-21--00.00
py.arg('--print_images', type=bool, default=True)
py.arg('--crop_size', type=int, default=256)  # Mura: 512 H2Z: 256
py.arg('--counterfactuals_type', type=str, default="abc-gan", choices=["abc-gan", "ganterfactual", "none"])
py.arg('--save_img', type=bool, default=True)
py.arg('--save_only_translated_img', type=bool, default=False)
py.arg('--tcv_os', type=bool, default=True)
py.arg('--ssim_psnr', type=bool, default=True)
py.arg('--kid', type=bool, default=True)

args = py.args()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
if len(tf.config.list_physical_devices('GPU')) == 0:
    TFDS_PATH = "/Users/dimitrymindlin/tensorflow_datasets"
else:
    TFDS_PATH = "../tensorflow_datasets"

# TFDS_PATH = "../tensorflow_datasets"

if args.dataset == "mura":
    args.crop_size = 512
    args.oracle_name = "densenet"
    args.oracle_ckp_name = "2022-08-15--17.42"
    args.clf_ckp_name = "2022-06-04--00.05"


def get_abc_gan_generators(name, ep):
    if name == "2022-05-23--18.32":  # normal MURA gan without attention
        tl.Checkpoint(dict(generator_g=G_A2B, generator_f=G_B2A),
                      py.join(f"{ROOT_DIR}/checkpoints/gans/{args.dataset}/{name}")).restore(
            save_path=f'{ROOT_DIR}/checkpoints/gans/{args.dataset}/{name}/ckpt-{ep}')
    else:
        tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A),
                      py.join(f"{ROOT_DIR}/checkpoints/gans/{args.dataset}/{name}")).restore(
            save_path=f'{ROOT_DIR}/checkpoints/gans/{args.dataset}/{name}/ckpt-{ep}')
    return G_A2B, G_B2A


def get_ganterfactual_generators(name, ep):
    cyclegan_folder = f"{ROOT_DIR}/checkpoints/gans/mura/{name}/{ep}"
    custom_objects = {"InstanceNormalization": InstanceNormalization}
    G_A2B = tf.keras.models.load_model(os.path.join(cyclegan_folder, 'generator_np.h5'),
                                       custom_objects=custom_objects)
    G_B2A = tf.keras.models.load_model(os.path.join(cyclegan_folder, 'generator_pn.h5'),
                                       custom_objects=custom_objects)
    return G_A2B, G_B2A


# ==============================================================================
# =                                    data                                    =
# ==============================================================================
if args.dataset == "mura":
    A_dataset, B_dataset, A_dataset_test, B_dataset_test = get_mura_test_ds_by_body_part_split_class(args.body_parts,
                                                                                                     TFDS_PATH,
                                                                                                     args.batch_size,
                                                                                                     args.crop_size,
                                                                                                     args.crop_size,
                                                                                                     special_normalisation=None)

else:  # Horse2Zebra / Apple2Orange
    A_dataset, A_dataset_test, B_dataset, B_dataset_test = load_tfds_test_data(args.dataset)

# ==============================================================================
# =                                    models                                  =
# ==============================================================================
G_A2B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))
G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))

clf = tf.keras.models.load_model(
    f"{ROOT_DIR}/checkpoints/{args.clf_name}_{args.dataset}/{args.clf_ckp_name}/model", compile=False)

"""oracle = tf.keras.models.load_model(
    f"{ROOT_DIR}/checkpoints/{args.oracle_name}_{args.dataset}/{args.oracle_ckp_name}/model", compile=False)"""

gradcam = GradcamPlusPlus(clf, clone=True)

"""done_h2z = ["2022-05-31--14.02", "2022-05-31--13.04", "2022-06-01--13.06", "2022-06-02--12.45"]
done_ep_h2z = ["180", "180", "180", "180"]
checkpoint_ts_list = ["2022-05-31--13.04", "2022-05-31--14.02", "2022-06-01--13.06", "2022-06-02--12.45",
                      "2022-06-03--14.07", "2022-06-03--19.10"]


checkpoint_ts_list_h2z = ["2022-08-13--15.48"]  # "2022-08-17--03.54"
checkpoint_ep_list_h2z = ["195"]  # 180"""

checkpoint_ts_list_abc = ["2022-09-23--16.36"]
checkpoint_ep_list_abc = ["180", "195"]

checkpoint_ts_list_ganterfactual = ["GANterfactual_2022-08-22--09.39", "GANterfactual_2022-08-31--08.19",
                                    "GANterfactual_2022-08-31--08.19"]
checkpoint_ep_list_ganterfactual = ["ep_16", "ep_14", "ep_17"]

checkpoint_ts_list_cyclegan = ["2022-08-29--12.05"]
checkpoint_ep_list_cyclegan = ["14"]


def load_generators_and_ckp_lists(counterfactuals_type):
    if counterfactuals_type == "abc-gan":
        load_generators = get_abc_gan_generators
        checkpoint_ts_list = checkpoint_ts_list_abc
        checkpoint_ep_list = checkpoint_ep_list_abc
    elif counterfactuals_type == "ganterfactual":
        load_generators = get_ganterfactual_generators
        checkpoint_ts_list = checkpoint_ts_list_ganterfactual
        checkpoint_ep_list = checkpoint_ep_list_ganterfactual
    else:  # CycleGAN
        load_generators = get_abc_gan_generators
        checkpoint_ts_list = checkpoint_ts_list_cyclegan
        checkpoint_ep_list = checkpoint_ep_list_cyclegan
    return load_generators, checkpoint_ts_list, checkpoint_ep_list


def evaluate_current_model(G_A2B, G_B2A, save_img=False):
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
        y_pred_translated, len_dataset, translated_images = translate_images_clf(
            source_dataset, clf, generator, gradcam, class_label, True, args.attention_type,
            training=False, save_img=save_img, save_only_translated_img=args.save_only_translated_img)

        if args.tcv_os:
            calculate_tcv(y_pred_translated, len_dataset, translation_name)

        if args.ssim_psnr:
            calculate_ssim_psnr(source_dataset, translated_images)

        if args.kid:
            if args.dataset == "mura":
                calc_KID_for_model(translated_images, args.crop_size, target_dataset)
            else:
                calc_KID_for_model_target_source(translated_images, translation_name, args.crop_size, A_dataset,
                                                 B_dataset)
    print()


counterfactuals_to_test = ["abc-gan"]
for counterfactuals_type in tqdm.tqdm(counterfactuals_to_test, desc='Counterfactual Type Loop'):
    with open(f'{counterfactuals_type}_{args.dataset}.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        load_generators, checkpoint_ts_list, checkpoint_ep_list = load_generators_and_ckp_lists(counterfactuals_type)
        for name, ep in zip(checkpoint_ts_list, checkpoint_ep_list):
            print(f"Starting {name}_{ep}")
            G_A2B, G_B2A = load_generators(name, ep)
            if args.save_img:
                save_img = name + "_" + ep
            else:
                save_img = False
            evaluate_current_model(G_A2B, G_B2A, save_img)
