import numpy as np
import tensorflow as tf
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from skimage.metrics import structural_similarity

import pylib as py
from config import ROOT_DIR
from evaluation.utils.load_test_data import load_test_data_from_args
from evaluation.utils.load_testing_models import load_models_for_testing
from test_config import config

# ==============================================================================
# =                                   param                                    =
# ==============================================================================
py.arg('--dataset', default='mura', choices=['horse2zebra', 'mura', 'apple2orange', 'rsna'])
py.arg('--save_img', type=bool, default=True)
py.arg('--body_parts', default=["XR_WRIST"])
py.arg('--generator', type=str, default="resnet", choices=['resnet', 'unet'])

test_args = py.args()
experiments_dir = f"{ROOT_DIR}/checkpoints/gans/{test_args.dataset}/"  # CycleGAN experiment results folder
TFDS_PATH = f"{ROOT_DIR}/../tensorflow_datasets"  # Path to datasets
SAVE_IMG = test_args.save_img
print(TFDS_PATH)

epsilon_values = [0.007, 0.05, 0.1]


def apply_perturbations(img, model, eps=0.1):
    """
    Applys FGSM to an image and returns the heatmap as well as the enhanced img.
    """
    img_fgsm = fast_gradient_method(model, img, eps, np.inf)
    img_fgsm = tf.clip_by_value(img_fgsm, -1, 1)
    return img_fgsm


def calc_img_difference(original_img, perturbed_img):
    """
    Use SSIM to calc img difference.
    """
    return structural_similarity(original_img, perturbed_img, channel_axis=2)


### Iterate over Models
for model_idx, (name, ep) in enumerate(
        zip(config[test_args.dataset]["model_names"], config[test_args.dataset]["epochs"])):
    # Load args from trained model
    test_args = py.args()
    args = py.load_args(py.join(experiments_dir, name), test_args=test_args)
    args.save_img = SAVE_IMG

    # Load all models
    G_A2B, G_B2A, clf, gradcam = load_models_for_testing(name, ep, args)

    # Run evaluations and save to file
    print(f"Starting {name}_{ep}")
    results_dict = {'Name': name,
                    'Epoch': ep}
    try:
        # Check if data already loaded
        print(len(A_dataset))
    except NameError:
        A_dataset, A_dataset_test, B_dataset, B_dataset_test = load_test_data_from_args(args)
        B_dataset_test.shuffle(buffer_size=500)
        # ds_to_eval = B_dataset_test.take(20).repeat(len(config[test_args.dataset]["model_names"]))
        ds_to_eval = B_dataset_test.skip(70).repeat(len(config[test_args.dataset]["model_names"]))
    ### Generate perturbations
    for img in ds_to_eval:
        for epsilons in epsilon_values:
            # Apply Perturbations
            img_fgsm = apply_perturbations(img, clf, eps=epsilons)

            # Calc difference
            diff = calc_img_difference(img, img_fgsm)

            # Save to dict
            results_dict[f"FGSM_{epsilons}"] = diff
