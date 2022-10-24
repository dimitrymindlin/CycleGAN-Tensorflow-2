from datetime import datetime
import time
import pylib as py
import tensorflow as tf
from train import run_training
import os

### Define Experiment Settings
py.arg('--dataset', default='rsna', choices=['horse2zebra', 'mura', 'apple2orange', "rsna"])
py.arg('--body_parts', default=["XR_WRIST"])  # Only used in Mura dataset. Body part of x-ray images
py.arg('--datasets_dir', default='datasets')
py.arg('--load_size', type=int, default=286)  # load image to this size
py.arg('--crop_size', type=int, default=256)  # then crop to this size
py.arg('--img_channels', type=int, default=3)
py.arg('--img_shape', type=tuple, default=(256, 256, 3))
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=200)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--adversarial_loss_mode', default='gan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--adversarial_loss_weight', type=float, default=1)
py.arg('--cycle_loss_weight', type=float, default=10)
py.arg('--counterfactual_loss_weight', type=float, default=0)
py.arg('--identity_loss_weight', type=float, default=0)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
py.arg('--attention', type=str, default="gradcam-plus-plus", choices=['gradcam', 'gradcam-plus-plus'])
py.arg('--clf_name', type=str, default="inception")
py.arg('--clf_ckp_name', type=str, default="2022-06-04--00.00")  # Mura: 2022-06-04--00.05, H2Z: 2022-06-04--00.00 A2O: 2022-09-23--15.18
py.arg('--attention_type', type=str, default="attention-gan-original",
       choices=['attention-gan-foreground', 'none', 'attention-gan-original'])
py.arg('--current_attention_type', type=str, default="none")
py.arg('--generator', type=str, default="resnet", choices=['resnet', 'unet'])
py.arg('--discriminator', type=str, default="patch-gan", choices=['classic', 'patch-gan', 'patch_gan_attention'])
py.arg('--disc_norm', type=str, default="instance_norm", choices=['instance_norm', 'none', 'batch_norm', 'layer_norm'])
py.arg('--load_checkpoint', type=str, default=None)
py.arg('--start_attention_epoch', type=int, default=0)
py.arg('--sample_interval', type=int, default=5)
args = py.args()

print(f"Using {args.dataset} dataset.")
if args.dataset == "mura":
    args.load_size = 512
    args.crop_size = 512
    args.epochs = 19
    args.epoch_decay = 16
    args.sample_interval = 2
    args.clf_ckp_name = "2022-06-04--00.05"

elif args.dataset == "rsna":
    args.load_size = 512
    args.crop_size = 512
    args.epochs = 19
    args.epoch_decay = 16
    args.sample_interval = 2
    # args.clf_ckp_name = "2022-10-12--10.37" # Inception
    if args.clf_name == "alexnet":
        args.clf_ckp_name = "2022-10-13--13.03"  # alexnet
        args.img_channels = 1
    if args.clf_name == "inception":
        args.clf_ckp_name = "2022-10-12--10.37" # inception
        args.img_channels = 3
elif args.dataset == "a2o":
    args.clf_ckp_name = "2022-09-23--15.18"
else: # h2z
    args.clf_ckp_name = "2022-06-04--00.00"

# Create new output dir if new experiment
if not args.load_checkpoint:
    execution_id = datetime.now().strftime("%Y-%m-%d--%H.%M")
    # output_dir
    try:
        output_dir = py.join(f'output_{args.dataset}/{execution_id}')
        py.mkdir(output_dir)
    except FileExistsError:
        time.sleep(60)
        execution_id = datetime.now().strftime("%Y-%m-%d--%H.%M")
        output_dir = py.join(f'output_{args.dataset}/{execution_id}')
        py.mkdir(output_dir)
else:
    # For loading checkpoint if continuation of experiment
    print(f"Setting {args.load_checkpoint} as checkpoint.")
    execution_id = args.load_checkpoint
    output_dir = py.join(f'output_{args.dataset}/{execution_id}')

# Make sure settings fit
if args.discriminator == "patch_gan_attention":
    # Remove instance norm as suggested in 'Unsupervised Attention-guided Image-to-Image Translation'
    args.disc_norm = "none"

TF_LOG_DIR = f"logs/{args.dataset}/"
if len(tf.config.list_physical_devices('GPU')) == 0:
    TFDS_PATH = "/Users/dimitrymindlin/tensorflow_datasets"
else:
    TFDS_PATH = "../tensorflow_datasets"
    #TFDS_PATH = "/Users/dimitrymindlin/tensorflow_datasets"
    #os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

### Start Experiment
run_training(args, TFDS_PATH, TF_LOG_DIR, output_dir, execution_id)
