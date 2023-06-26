import pylib as py
from config import ROOT_DIR
from evaluation.plots.generate_img_grid import generate_images_for_grid
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
SAVE_IMG = test_args.save_img


def get_save_img(args):
    """
    Sets and returns the variable save_img that determines if images should be saved and if so, where.
    """
    if args.save_img:
        if args.dataset == "apple2orange":
            save_img = py.join(experiments_dir, name, f"test_images_target_source_{ep}")
        else:
            save_img = py.join(experiments_dir, name, f"test_images_{ep}")
    else:
        save_img = False
    return save_img


axs = None
# Iterate over model name and saved epoch
for model_idx, (name, ep) in enumerate(
        zip(config[test_args.dataset]["model_names"], config[test_args.dataset]["epochs"])):
    # Load args from trained model
    test_args = py.args()
    args = py.load_args(py.join(experiments_dir, name), test_args=test_args)
    args.save_img = SAVE_IMG

    # Load all models
    G_A2B, G_B2A, clf, gradcam = load_models_for_testing(name, ep, args)

    # Run evaluations and save to file
    with open(py.join(experiments_dir, name, f'test_metrics_{ep}.txt'), 'w') as f:
        # sys.stdout = f  # Change the standard output to the file we created.
        print(f"Starting {name}_{ep}")
        try:
            # Check if data already loaded
            print(len(A_dataset))
        except NameError:
            A_dataset, A_dataset_test, B_dataset, B_dataset_test = load_test_data_from_args(args)
            B_dataset_test.shuffle(buffer_size=500)
            # ds_to_eval = B_dataset_test.take(20).repeat(len(config[test_args.dataset]["model_names"]))
            ds_to_eval = B_dataset_test.skip(180).repeat(len(config[test_args.dataset]["model_names"]))
        save_img = get_save_img(args)
        # Load images to tensors from folder
        """images = []
        for img_path in glob.glob("/Users/dimitrymindlin/UniProjects/CycleGAN-Tensorflow-2/b_images/*.png"):
            img = load_img(img_path, target_size=(512, 512))
            img_array = np.array(img)
            img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
            img_tensor = tf.expand_dims(img_tensor, axis=0)
            img_tensor = scale_between_minus_one_one(img_tensor)
            images.append(img_tensor)
"""
        axs = generate_images_for_grid(model_idx, args, ds_to_eval, clf, G_B2A, gradcam, 1, training=False,
                                       num_images=6, axs=axs)
