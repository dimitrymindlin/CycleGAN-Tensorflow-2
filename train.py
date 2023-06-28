import functools

from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import tqdm

import data
import module

from imlib.image_segmentation import get_img_segmentations
from cyclegan_strategies.abc_gan import abc_gan_step, abc_gan_discriminator_step
from cyclegan_strategies.cyclegan import cycleGAN_step
from mura import get_mura_ds_by_body_part_split_class
from dataset_configs.mura_config import config as mura_config
from rsna import get_rsna_ds_split_class
from global_config import ROOT_DIR

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--dataset', default='mura')
py.arg('--datasets_dir', default='datasets')
py.arg('--load_size', type=int, default=512)  # load image to this size
py.arg('--crop_size', type=int, default=512)  # then crop to this size
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=200)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--cycle_loss_weight', type=float, default=10.0)
py.arg('--counterfactual_loss_weight', type=float, default=0.0)
py.arg('--identity_loss_weight', type=float, default=0.0)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
py.arg('--cyclegan_mode', default='abc-gan', choices=['cyclegan', 'abc-gan', 'ganterfactual'])
py.arg('--clf_name', default='inception', choices=['inception', 'alexnet'])
py.arg('--clf_ckp_name', default=None)  # checkpoint name of the classifier to load
py.arg('--start_attention_epoch', default=0, type=int)  # epoch to start using attention maps

args = py.args()

# TODO: (Dimi) Make this better....
args.clf_ckp_name = "2022-06-04--00.05"
TFDS_PATH = ROOT_DIR + "/../tensorflow_datasets"
# output_dir
output_dir = py.join('output', args.dataset)
py.mkdir(output_dir)

# ==============================================================================
# =                                    data                                    =
# ==============================================================================

if args.clf_name in ["alexnet", "inception"]:
    special_normalisation = tf.keras.applications.inception_v3.preprocess_input
else:
    special_normalisation = None

if args.dataset == "mura":
    # A = Normal, B = Abnormal
    A_B_dataset, A_B_dataset_valid, A_B_dataset_test, len_dataset = get_mura_ds_by_body_part_split_class(
        mura_config['body_part'],
        TFDS_PATH,
        args.batch_size,
        mura_config['img_width'],
        mura_config['img_width'],
        special_normalisation)
elif args.dataset == "rsna":
    A_B_dataset, A_B_dataset_valid, A_B_dataset_test, len_dataset = get_rsna_ds_split_class(TFDS_PATH,
                                                                                            args.batch_size,
                                                                                            args.crop_size,
                                                                                            args.load_size,
                                                                                            special_normalisation,
                                                                                            channels=args.img_channels)
else:  # All CycleGAN datasets.
    A_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, 'trainA'), '*.jpg')
    B_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, 'trainB'), '*.jpg')
    A_B_dataset, len_dataset = data.make_zip_dataset(A_img_paths, B_img_paths, args.batch_size, args.load_size,
                                                     args.crop_size, training=True, repeat=False)
    A_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testA'), '*.jpg')
    B_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testB'), '*.jpg')
    A_B_dataset_test, _ = data.make_zip_dataset(A_img_paths_test, B_img_paths_test, args.batch_size, args.load_size,
                                                args.crop_size, training=False, repeat=True)

A2B_pool = data.ItemPool(args.pool_size)
B2A_pool = data.ItemPool(args.pool_size)

# ==============================================================================
# =                                   models                                   =
# ==============================================================================

G_A2B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))
G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))

D_A = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, 3))
D_B = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, 3))

d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)
cycle_loss_fn = tf.losses.MeanAbsoluteError()
identity_loss_fn = tf.losses.MeanAbsoluteError()
counterfactual_loss_fn = tf.losses.MeanSquaredError()  # Counterfactual loss originally from https://github.com/hcmlab/GANterfactual
# Ground truth for counterfactual loss
class_A_ground_truth = np.stack([np.ones(args.batch_size), np.zeros(args.batch_size)]).T
class_B_ground_truth = np.stack([np.zeros(args.batch_size), np.ones(args.batch_size)]).T

# Optimizers
G_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
D_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1)
D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1)

# Dicscriminator Accuracy Metrics
train_D_A_acc = tf.keras.metrics.BinaryAccuracy()
train_D_B_acc = tf.keras.metrics.BinaryAccuracy()

# Set classifier and gradcam if ABC-GAN or GANterfactual
# Check if it's a normal run
if not args.cyclegan_mode == "cyclegan":
    clf = None
    gradcam = None

if args.cyclegan_mode in ["ganterfactual", "abc-gan"]:
    # Load the classifier model
    model_path = f"{ROOT_DIR}/checkpoints/{args.clf_name}_{args.dataset}/{args.clf_ckp_name}/model"
    clf = tf.keras.models.load_model(model_path, compile=False)
    # Get the input channel of the classifier
    args.clf_input_channel = clf.layers[0].input_shape[0][-1]
    if args.cyclegan_mode == "abc-gan":  # abc-gan
        gradcam = GradcamPlusPlus(clf, clone=True)

# save settings.yml here, since args.clf_input_channel might be set before
settings_name = "settings.yml"
py.args_to_yaml(py.join(output_dir, settings_name), args)


# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G_no_attention(A, B):
    with tf.GradientTape() as t:
        A2B, B2A, A2B2A, B2A2B, A2A, B2B = cycleGAN_step(A, B, G_A2B, G_B2A)

        A2B_d_logits = D_B(A2B, training=True)
        B2A_d_logits = D_A(B2A, training=True)

        A2B_g_loss = g_loss_fn(A2B_d_logits)
        B2A_g_loss = g_loss_fn(B2A_d_logits)
        A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)
        B2A2B_cycle_loss = cycle_loss_fn(B, B2A2B)
        A2A_id_loss = identity_loss_fn(A, A2A)
        B2B_id_loss = identity_loss_fn(B, B2B)

        # Add counterfactual loss for GANterfactual
        if args.counterfactual_loss_weight > 0:
            if args.clf_input_channel == 1:
                A2B_clf = tf.image.rgb_to_grayscale(A2B)
                B2A_clf = tf.image.rgb_to_grayscale(B2A)
            else:
                A2B_clf = A2B
                B2A_clf = B2A
            A2B_counterfactual_loss = counterfactual_loss_fn(class_B_ground_truth,
                                                             clf(tf.image.resize(A2B_clf, [512, 512],
                                                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)))
            B2A_counterfactual_loss = counterfactual_loss_fn(class_A_ground_truth,
                                                             clf(tf.image.resize(B2A_clf, [512, 512],
                                                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)))

        else:
            A2B_counterfactual_loss = tf.zeros(())
            B2A_counterfactual_loss = tf.zeros(())

        G_loss = (A2B_g_loss + B2A_g_loss) + (A2B2A_cycle_loss + B2A2B_cycle_loss) * args.cycle_loss_weight + (
                A2A_id_loss + B2B_id_loss) * args.identity_loss_weight + \
                 (A2B_counterfactual_loss + B2A_counterfactual_loss) * args.counterfactual_loss_weight

    G_grad = t.gradient(G_loss, G_A2B.trainable_variables + G_B2A.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables + G_B2A.trainable_variables))

    return A2B, B2A, {'A2B_g_loss': A2B_g_loss,
                      'B2A_g_loss': B2A_g_loss,
                      'A2B2A_cycle_loss': A2B2A_cycle_loss,
                      'B2A2B_cycle_loss': B2A2B_cycle_loss,
                      'A2A_id_loss': A2A_id_loss,
                      'B2B_id_loss': B2B_id_loss}


@tf.function
def train_G_ABC_GAN(A_img, B_img, A_attention, B_attention, A_background, B_background):
    training = True
    with tf.GradientTape() as t:
        # Generate Images based on attention-gan strategy
        A2B, B2A, A2B2A, B2A2B, A2A, B2B = abc_gan_step(A_img, B_img, G_A2B, G_B2A, A_attention, B_attention,
                                                        A_background, B_background, training)
        # Calculate Losses
        A2B_d_logits = D_B(A2B, training=training)
        B2A_d_logits = D_A(B2A, training=training)

        if args.clf_input_channel == 1:
            A2B = tf.image.rgb_to_grayscale(A2B)
            B2A = tf.image.rgb_to_grayscale(B2A)

        # Calc counterfactual loss
        A2B_counterfactual_loss = counterfactual_loss_fn(class_B_ground_truth,
                                                         clf(tf.image.resize(A2B, [512, 512],
                                                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)))
        B2A_counterfactual_loss = counterfactual_loss_fn(class_A_ground_truth,
                                                         clf(tf.image.resize(B2A, [512, 512],
                                                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)))

        A2B_g_loss = g_loss_fn(A2B_d_logits)
        B2A_g_loss = g_loss_fn(B2A_d_logits)
        A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)
        B2A2B_cycle_loss = cycle_loss_fn(B, B2A2B)
        A2A_id_loss = identity_loss_fn(A, A2A)
        B2B_id_loss = identity_loss_fn(B, B2B)

        # Weighted Generator Loss
        G_loss = (A2B_g_loss + B2A_g_loss) + \
                 (A2B2A_cycle_loss + B2A2B_cycle_loss) * args.cycle_loss_weight \
                 + (A2A_id_loss + B2B_id_loss) * args.identity_loss_weight + \
                 (A2B_counterfactual_loss + B2A_counterfactual_loss) * args.counterfactual_loss_weight

        G_loss_dict = {'A2B_g_loss': A2B_g_loss,
                       'B2A_g_loss': B2A_g_loss,
                       'A2B2A_cycle_loss': A2B2A_cycle_loss,
                       'B2A2B_cycle_loss': B2A2B_cycle_loss,
                       'A2A_id_loss': A2A_id_loss,
                       'B2B_id_loss': B2B_id_loss,
                       'A2B_counterfactual_loss': A2B_counterfactual_loss,
                       'B2A_counterfactual_loss': B2A_counterfactual_loss}

    G_grad = t.gradient(G_loss, G_A2B.trainable_variables + G_B2A.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables + G_B2A.trainable_variables))
    return A2B, B2A, G_loss_dict


@tf.function
def train_D(A, B, A2B, B2A):
    # Reset Accuracy metrics for new batch.
    train_D_A_acc.reset_states()
    train_D_B_acc.reset_states()

    with tf.GradientTape() as t:
        A_d_logits = D_A(A, training=True)
        B2A_d_logits = D_A(B2A, training=True)
        B_d_logits = D_B(B, training=True)
        A2B_d_logits = D_B(A2B, training=True)

        A_d_loss, B2A_d_loss = d_loss_fn(A_d_logits, B2A_d_logits)
        B_d_loss, A2B_d_loss = d_loss_fn(B_d_logits, A2B_d_logits)
        D_A_gp = gan.gradient_penalty(functools.partial(D_A, training=True), A, B2A,
                                      mode=args.gradient_penalty_mode)
        D_B_gp = gan.gradient_penalty(functools.partial(D_B, training=True), B, A2B,
                                      mode=args.gradient_penalty_mode)
        D_loss = (A_d_loss + B2A_d_loss) + (B_d_loss + A2B_d_loss) + (
                D_A_gp + D_B_gp) * args.gradient_penalty_weight
        D_grad = t.gradient(D_loss, D_A.trainable_variables + D_B.trainable_variables)
        D_optimizer.apply_gradients(zip(D_grad, D_A.trainable_variables + D_B.trainable_variables))

        # Update training metric.
        train_D_A_acc.update_state(tf.ones_like(A_d_logits), A_d_logits)
        train_D_A_acc.update_state(tf.ones_like(B2A_d_logits), B2A_d_logits)
        train_D_B_acc.update_state(tf.zeros_like(B_d_logits), B_d_logits)
        train_D_B_acc.update_state(tf.zeros_like(A2B_d_logits), A2B_d_logits)

        return {'A_d_loss': A_d_loss + B2A_d_loss,
                'B_d_loss': B_d_loss + A2B_d_loss,
                'D_A_gp': D_A_gp,
                'D_B_gp': D_B_gp,
                'D_A_acc': train_D_A_acc.result(),
                'D_B_acc': train_D_B_acc.result()}


def train_step(A, B):
    A2B, B2A, G_loss_dict = train_G_no_attention(A, B)

    # cannot autograph `A2B_pool`
    A2B = A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
    B2A = B2A_pool(B2A)  # because of the communication between CPU and GPU

    D_loss_dict = train_D(A, B, A2B, B2A)

    return G_loss_dict, D_loss_dict


def train_step_ganterfactual(A, B):
    return train_step(A, B)  # TODO: Split from normal cyclegan? Make classes?


def train_step_ABC_GAN(A_img_seg, B_img_seg):
    A2B, B2A, G_loss_dict = train_G_ABC_GAN(A_img_seg.img, B_img_seg.img,
                                            A_img_seg.attention, B_img_seg.attention,
                                            A_img_seg.background, B_img_seg.background)

    # cannot autograph `A2B_pool`
    A2B = A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
    B2A = B2A_pool(B2A)  # because of the communication between CPU and GPU

    if args.discriminator == "patch_gan_attention":
        # Attentive discriminator mode: use attention map to guide the discriminator
        # Here, use attention map on source and target images to only give foreground to the discriminator
        A, A2B = abc_gan_discriminator_step(A_img_seg.img, A2B, A_img_seg.attention)
        B, B2A = abc_gan_discriminator_step(B_img_seg.img, B2A, B_img_seg.attention)
    else:
        # If not attentive discriminator, use the whole image
        A = A_img_seg.img
        B = B_img_seg.img

    D_loss_dict = train_D(A, B, A2B, B2A)
    return G_loss_dict, D_loss_dict


@tf.function
def sample(A, B):
    A2B = G_A2B(A, training=False)
    B2A = G_B2A(B, training=False)
    A2B2A = G_B2A(A2B, training=False)
    B2A2B = G_A2B(B2A, training=False)
    return A2B, B2A, A2B2A, B2A2B


@tf.function
def sample_abc_gan(A_img_seg, B_img_seg):
    training = False
    # Generate Images based on abc-gan strategy
    A2B_transformed, B2A_transformed, _, _ = abc_gan_step(A_img_seg.img,
                                                          B_img_seg.img, G_A2B, G_B2A,
                                                          A_img_seg.attention,
                                                          B_img_seg.attention,
                                                          A_img_seg.background,
                                                          B_img_seg.background,
                                                          training)
    return A2B_transformed, B2A_transformed


# ==============================================================================
# =                                   helper funcs                             =
# ==============================================================================
def set_current_attention_type(args, ep):
    """Method checks if attention should be used in the current epoch."""
    if ep < args.start_attention_epoch:
        args.current_attention_type = "none"
    else:
        args.current_attention_type = "attention"


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.utils.Checkpoint(dict(G_A2B=G_A2B,
                                      G_B2A=G_B2A,
                                      D_A=D_A,
                                      D_B=D_B,
                                      G_optimizer=G_optimizer,
                                      D_optimizer=D_optimizer,
                                      ep_cnt=ep_cnt),
                                 py.join(output_dir, 'checkpoints'),
                                 max_to_keep=5)
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'train'))

# sample
test_iter = iter(A_B_dataset_test)
sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)

# main loop
with train_summary_writer.as_default():
    for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue

        # update epoch counter
        ep_cnt.assign_add(1)

        # train for an epoch
        for A, B in tqdm.tqdm(A_B_dataset, desc='Inner Epoch Loop', total=len_dataset):
            if args.cyclegan_mode == "cyclegan":
                G_loss_dict, D_loss_dict = train_step(A, B)
            elif args.cyclegan_mode == "ganterfactual":
                G_loss_dict, D_loss_dict = train_step_ganterfactual(A, B)
            else:  # ABC-GAN
                set_current_attention_type(args, ep)  # important if attention should start later than epoch 0
                if args.current_attention_type == "attention":
                    A_img_seg, B_img_seg = get_img_segmentations(A, B, args)
                    G_loss_dict, D_loss_dict = train_step_ABC_GAN(A_img_seg, B_img_seg)
                else:
                    # Without attention, use ganterfactual approach
                    G_loss_dict, D_loss_dict = train_step_ganterfactual(A, B)

            # # summary
            tl.utils.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.utils.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
            tl.utils.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations,
                             name='learning rate')

            # sample and save img every 100 steps
            if G_optimizer.iterations.numpy() % 100 == 0:
                A, B = next(test_iter)
                if args.cyclegan_mode == "cyclegan" or args.cyclegan_mode == "ganterfactual":
                    A2B, B2A, A2B2A, B2A2B = sample(A, B)
                    img = im.immerge(np.concatenate([A, A2B, A2B2A, B, B2A, B2A2B], axis=0), n_rows=2)
                    im.imwrite(img, py.join(sample_dir, 'iter-%09d.jpg' % G_optimizer.iterations.numpy()))
                else:  # ABC-GAN
                    A2B_transformed, B2A_transformed = sample_abc_gan(A_img_seg, B_img_seg)
                    img = im.immerge(np.concatenate(
                        [A, A_img_seg.attention, A2B_transformed, B, B_img_seg.attention, B2A_transformed], axis=0),
                        n_rows=2)
                    im.imwrite(img, py.join(sample_dir, 'iter-%09d.jpg' % G_optimizer.iterations.numpy()))

        # save checkpoint
        checkpoint.save(ep)