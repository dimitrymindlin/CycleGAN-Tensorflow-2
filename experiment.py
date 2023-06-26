from datetime import datetime
import time
import pylib as py
from config import ROOT_DIR
from pylib import load_args
from train import run_training


def create_new_experiment(root_dir):
    # Create new experiment and dirs
    execution_id = datetime.now().strftime("%Y-%m-%d--%H.%M")
    # output_dir
    try:
        output_dir = py.join(f"{root_dir}/checkpoints/gans/{args.dataset}/{execution_id}")
    except FileExistsError:
        time.sleep(60)
        execution_id = datetime.now().strftime("%Y-%m-%d--%H.%M")
        output_dir = py.join(f"{root_dir}/checkpoints/gans/{args.dataset}/{execution_id}")
    py.mkdir(output_dir)
    return output_dir, execution_id


def setup_args_for_experiment():
    if args.dataset == "mura":
        args.load_size = 512
        args.crop_size = 512
        args.epochs = 19
        args.epoch_decay = 16
        args.sample_interval = 2
        args.clf_ckp_name = "2022-06-04--00.05"
        args.clf_name = "inception"

    elif args.dataset == "rsna":
        args.load_size = 512
        args.crop_size = 512
        args.epochs = 19
        args.epoch_decay = 16
        args.sample_interval = 2
        # args.clf_ckp_name = "2022-10-12--10.37" # Inception
        if args.clf_name == "alexnet":
            args.clf_ckp_name = "2022-10-13--13.03"
        if args.clf_name == "inception":
            args.clf_ckp_name = "2022-10-12--10.37"
    elif args.dataset == "apple2orange":
        args.clf_ckp_name = "2023-04-17--15.53"  # 2022-09-23--15.18
        args.clf_name = "simplenet"
        args.crop_size = 256
        # h2z "2022-06-04--00.00"
    elif args.dataset == "celeba":
        args.crop_size = 218
        args.clf_ckp_name = "2023-04-20--12.00"  # 2022-09-23--15.18
        args.clf_name = "inception"

    if args.discriminator == "patch_gan_attention":
        # Remove instance norm as suggested in 'Unsupervised Attention-guided Image-to-Image Translation'
        args.disc_norm = "none"
    return args


### Define Experiment Settings
py.arg('--dataset', default='celeba', choices=['horse2zebra', 'mura', 'apple2orange', "rsna", "celeba"])
py.arg('--body_parts', default=["XR_WRIST"])  # Only used in Mura dataset. Body part of x-ray images
py.arg('--datasets_dir', default='datasets')
py.arg('--load_size', type=int, default=218)  # load image to this size
py.arg('--crop_size', type=int, default=218)  # then crop to this size
py.arg('--img_channels', type=int, default=3)
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=200)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--adversarial_loss_mode', default='gan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--adversarial_loss_weight', type=float, default=1)
py.arg('--cycle_loss_weight', type=float, default=10)
py.arg('--counterfactual_loss_weight', type=float, default=1)
py.arg('--identity_loss_weight', type=float, default=1)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
py.arg('--attention', type=str, default="gradcam-plus-plus", choices=['gradcam', 'gradcam-plus-plus', 'occlusion'])
py.arg('--clf_name', type=str, default="simplenet", choices=['alexnet', 'inception', 'simplenet'])
py.arg('--clf_ckp_name', type=str,
       default="2023-04-17--15.53")  # Mura: 2022-06-04--00.05, H2Z: 2022-06-04--00.00 A2O: 2022-09-23--15.18 RSNA: "2022-06-04--00.00 Celeba: 2023-04-16--17.02
py.arg('--attention_type', type=str, default="none",
       choices=['attention-gan-foreground', 'none', 'attention-gan-original'])
py.arg('--current_attention_type', type=str, default="none")
py.arg('--generator', type=str, default="resnet", choices=['resnet', 'unet'])
py.arg('--discriminator', type=str, default="patch_gan_attention",
       choices=['classic', 'patch-gan', 'patch_gan_attention'])
py.arg('--disc_norm', type=str, default="instance_norm", choices=['instance_norm', 'none', 'batch_norm', 'layer_norm'])
py.arg('--load_checkpoint', type=str, default=None)
py.arg('--start_attention_epoch', type=int, default=0)
py.arg('--sample_interval', type=int, default=5)
py.arg('--precompute_attention', type=bool, default=False)
args = py.args()

print(f"Using {args.dataset} dataset.")
if args.load_checkpoint:
    # For loading checkpoint if continuation of experiment
    print(f"Setting {args.load_checkpoint} as checkpoint.")
    execution_id = args.load_checkpoint
    output_dir = py.join(f'{ROOT_DIR}/checkpoints/gans/{args.dataset}/{execution_id}')
    new_epoch_count = args.epochs
    args = load_args(output_dir)
    args.epochs = new_epoch_count
else:
    output_dir, execution_id = create_new_experiment(ROOT_DIR)
    setup_args_for_experiment()

TF_LOG_DIR = f"logs/{args.dataset}/"
TFDS_PATH = f"{ROOT_DIR}/../tensorflow_datasets"

### Start Experiment
run_training(args, TFDS_PATH, TF_LOG_DIR, output_dir, execution_id)
