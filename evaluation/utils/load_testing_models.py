import os

import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

import module
import pylib as py
import tf2lib_local as tl
from config import ROOT_DIR
from tf2lib_local.utils import is_ganterfactual_repo


def load_models_for_testing(name, ep, args):
    """
    Loads and returns Counterfactual Model, clf and GradCAM as defined in args.
    """
    # Get Generators
    load_generators = get_load_generators(args)
    generator = module.ResnetGenerator if args.generator == "resnet" else module.UnetGenerator
    G_A2B = generator(input_shape=args.img_shape)
    G_B2A = generator(input_shape=args.img_shape)
    G_A2B, G_B2A = load_generators(G_A2B, G_B2A, name, ep, args)

    # Get CLF + Gradcam
    clf = tf.keras.models.load_model(
        f"{ROOT_DIR}/checkpoints/{args.clf_name}_{args.dataset}/{args.clf_ckp_name}/model", compile=False)

    gradcam = GradcamPlusPlus(clf, clone=True)

    # Set clf input shape if not done:
    try:
        args.clf_input_channel
    except AttributeError:
        args.clf_input_channel = clf.layers[0].input_shape[0][-1]

    return G_A2B, G_B2A, clf, gradcam


def get_load_generators(args):
    """
    Decide if generators from this project or GANterfactual project.
    """
    if is_ganterfactual_repo(args):
        load_generators = get_ganterfactual_generators
    else:
        load_generators = get_abc_gan_generators
    return load_generators


def get_abc_gan_generators(G_A2B, G_B2A, timestamp_id, ep, args):
    if timestamp_id == "2022-05-23--18.32":  # normal MURA gan without attention
        tl.utils.Checkpoint(dict(generator_g=G_A2B, generator_f=G_B2A),
                            py.join(f"{ROOT_DIR}/checkpoints/gans/{args.dataset}/{timestamp_id}")).restore(
            save_path=f'{ROOT_DIR}/checkpoints/gans/{args.dataset}/{timestamp_id}/ckpt-{ep}')
    else:
        if os.path.exists(py.join(f"{ROOT_DIR}/checkpoints/gans/{args.dataset}/{timestamp_id}/checkpoints")):
            # New runs will keep checkpoints in checkpoints folder
            tl.utils.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A),
                                py.join(
                                    f"{ROOT_DIR}/checkpoints/gans/{args.dataset}/{timestamp_id}/checkpoints")).restore(
                save_path=f'{ROOT_DIR}/checkpoints/gans/{args.dataset}/{timestamp_id}/checkpoints/ckpt-{ep}')
        else:
            tl.utils.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A),
                                py.join(f"{ROOT_DIR}/checkpoints/gans/{args.dataset}/{timestamp_id}")).restore(
                save_path=f'{ROOT_DIR}/checkpoints/gans/{args.dataset}/{timestamp_id}/ckpt-{ep}')
    return G_A2B, G_B2A


def get_ganterfactual_generators(G_A2B, G_B2A, name, ep, args):
    cyclegan_folder = f"{ROOT_DIR}/checkpoints/gans/{args.dataset}/{name}/{ep}"
    custom_objects = {"InstanceNormalization": InstanceNormalization}
    G_A2B = tf.keras.models.load_model(os.path.join(cyclegan_folder, 'generator_np.h5'),
                                       custom_objects=custom_objects)
    G_B2A = tf.keras.models.load_model(os.path.join(cyclegan_folder, 'generator_pn.h5'),
                                       custom_objects=custom_objects)
    return G_A2B, G_B2A
