from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from tqdm import tqdm

import pylib as py
from attention_strategies.attention_gan import attention_gan_single
from attention_strategies.no_attention import no_attention_single
from config import ROOT_DIR
from evaluation.metrics.calc_ssim import calc_ssim
from evaluation.utils.load_test_data import load_test_data_from_args
from evaluation.utils.load_testing_models import load_models_for_testing
from imlib.image_holder import ImageHolder
from test_config import config

# ==============================================================================
# =                                   param                                    =
# ==============================================================================
py.arg('--dataset', default='rsna', choices=['horse2zebra', 'mura', 'apple2orange', 'rsna'])
py.arg('--body_parts', default=["XR_WRIST"])
py.arg('--generator', type=str, default="resnet", choices=['resnet', 'unet'])

test_args = py.args()
ts = datetime.now().strftime("%Y-%m-%d--%H.%M")
experiments_dir = f"{ROOT_DIR}/checkpoints/gans/{test_args.dataset}/"  # CycleGAN experiment results folder
TFDS_PATH = f"{ROOT_DIR}/../tensorflow_datasets"  # Path to datasets

epsilon_values = [0.007, 0.05, 0.1]


def apply_perturbations(img, model, eps=0.1):
    """
    Applys FGSM to an image and returns the heatmap as well as the enhanced img.
    """

    img_tmp = tf.image.rgb_to_grayscale(img) if tf.shape(img)[-1] == 3 else img
    img_fgsm = fast_gradient_method(model, img_tmp, eps, np.inf)
    img_fgsm = tf.clip_by_value(img_fgsm, -1, 1)
    return img_fgsm


### Iterate over Models
results_table_dict = {}
training = False
for model_idx, (name, ep) in enumerate(tqdm(
        zip(config[test_args.dataset]["model_names"], config[test_args.dataset]["epochs"]))):
    results_dict = {}
    # Load args from trained model
    test_args = py.args()
    args = py.load_args(py.join(experiments_dir, name), test_args=test_args)

    # Load all models
    G_A2B, G_B2A, clf, gradcam = load_models_for_testing(name, ep, args)
    # Run evaluations and save to file
    print(f"Starting {name}_{ep}")
    use_attention = True if args.attention_type != "none" else False
    try:
        # Check if data already loaded
        print(len(A_dataset))
    except NameError:
        A_dataset, A_dataset_test, B_dataset, B_dataset_test = load_test_data_from_args(args)
        B_dataset_test.shuffle(buffer_size=500)
        # ds_to_eval = B_dataset_test.skip(70).repeat(len(config[test_args.dataset]["model_names"]))
    ### Calculate ssim for each test img
    for epsilon in epsilon_values:
        epsilon_results = []  # Split in A AND B results
        for dataset, generator, class_label in zip([A_dataset_test, B_dataset_test], [G_A2B, G_B2A], [0, 1]):
            for img in tqdm(dataset):
                # Apply Perturbations
                img_fgsm = apply_perturbations(img, clf, eps=epsilon)
                # Get img holders
                img_holder = ImageHolder(img, args, class_label, attention_func=gradcam, use_attention=use_attention)
                img_holder_fgsm = ImageHolder(img_fgsm, args, class_label, attention_func=gradcam, use_attention=use_attention)
                # Generate Counterfactuals
                if args.attention_type == "attention-gan-original":
                    translated_img, _ = attention_gan_single(img_holder.img, generator, None, img_holder.attention,
                                                             img_holder.background, training)
                    translated_img_fsgm, _ = attention_gan_single(img_holder_fgsm.img, generator, None,
                                                                  img_holder_fgsm.attention,
                                                                  img_holder_fgsm.background, training)
                else:
                    translated_img = no_attention_single(img_holder.img, generator, None, training)
                    if tf.shape(img_holder_fgsm.img)[-1] == 1:
                        img_holder_fgsm.img = tf.image.grayscale_to_rgb(img_holder_fgsm.img)
                    translated_img_fsgm = no_attention_single(img_holder_fgsm.img, generator, None, training)

                # Calc SSIM similarity
                diff_value = 1 - calc_ssim(translated_img, translated_img_fsgm)

                # Save to list
                epsilon_results.append(diff_value)
            # Average epsilon results and save to dict
            class_label_name = "A2B" if class_label == 0 else "B2A"
            results_dict[class_label_name + "_" + str(epsilon)] = np.sum(epsilon_results) / len(epsilon_results)
            print(f"{name}_{ep}_{epsilon} done")
        results_table_dict[name + "_" + ep] = results_dict
for epsilon in epsilon_values:
    for name, ep in zip(config[test_args.dataset]["model_names"], config[test_args.dataset]["epochs"]):
        results_table_dict[name + "_" + ep][f"AVG_{epsilon}"] = round(
            (results_table_dict[name + "_" + ep][f"A2B_{epsilon}"] +
             results_table_dict[name + "_" + ep][f"B2A_{epsilon}"]) / 2, ndigits=2)
results_pd = pd.DataFrame.from_dict(results_table_dict, orient='index')
print(results_pd.to_latex(index=True, index_names=True))
results_pd.to_csv(f"{experiments_dir}/robustness_results_{args.dataset}_{ts}.csv", index=True)
