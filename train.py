import functools
import os
import time
from datetime import datetime

from matplotlib import pyplot as plt

import attention_maps
import imlib as im
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import tqdm
import numpy as np
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.gradcam import Gradcam

import data
import module
from attention_strategies.attention_strategies import attention_gan_original, attention_gan_foreground
from imlib import save_mura_images, save_mura_images_with_attention, save_images, save_images_with_attention
from imlib.attention_image import AttentionImage, add_images

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--dataset', default='apple2orange')
py.arg('--datasets_dir', default='datasets')
py.arg('--load_size', type=int, default=520)  # load image to this size
py.arg('--crop_size', type=int, default=512)  # then crop to this size
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=30)
py.arg('--epoch_decay', type=int, default=15)  # epoch to start decaying learning rate
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
py.arg('--gradient_penalty_weight', type=float, default=1)
py.arg('--cycle_loss_weight', type=float, default=1)
py.arg('--counterfactual_loss_weight', type=float, default=1)
py.arg('--identity_loss_weight', type=float, default=0.0)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
"""pool_size: the discriminator is trained against the current batch of generated images as well as images generated on 
previous iterations. Essentially, we remember the last pool_size generated images then randomly sample from this pool 
to create a batch_size batch of images to do one iteration of backprop on. This helps to stabilize training, kind of 
like experience replay."""
py.arg('--attention', type=str, default="gradcam-plus-plus", choices=['gradcam', 'gradcam-plus-plus'])
py.arg('--attention_type', type=str, default="none", choices=['attention-gan', 'spa-gan', 'none'])
py.arg('--attention_intensity', type=float, default=0.5)
py.arg('--attention_gan_original', type=bool, default=True)
py.arg('--generator', type=str, default="resnet", choices=['resnet', 'unet'])
args = py.args()

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

TF_LOG_DIR = f"logs/{args.dataset}/"

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

# ==============================================================================
# =                                    data                                    =
# ==============================================================================
A_img_paths, B_img_paths, A_img_paths_test, B_img_paths_test = data.get_dataset_paths(args)
A_B_dataset, len_dataset = data.make_zip_dataset(A_img_paths, B_img_paths, args.batch_size, args.load_size,
                                                 args.crop_size, training=True, repeat=False)
A_B_dataset_test, _ = data.make_zip_dataset(A_img_paths_test, B_img_paths_test, args.batch_size, args.load_size,
                                            args.crop_size, training=False, repeat=True)

A2B_pool = data.ItemPool(args.pool_size)
B2A_pool = data.ItemPool(args.pool_size)

# ==============================================================================
# =                                   models                                   =
# ==============================================================================

G_A2B, G_B2A = module.get_generators(args)

D_A = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, 3))
D_B = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, 3))

d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)
cycle_loss_fn = tf.losses.MeanAbsoluteError()
identity_loss_fn = tf.losses.MeanAbsoluteError()
counterfactual_loss_nf = tf.losses.MeanSquaredError()

if args.dataset == "mura":
    clf = tf.keras.models.load_model(f"checkpoints/2022-03-24--12.42/model", compile=False)
else:
    clf = tf.keras.models.load_model(f"checkpoints/inception_{args.dataset}/model", compile=False)

G_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
D_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1)
D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1)
class_A_ground_truth = np.stack([np.ones(args.batch_size), np.zeros(args.batch_size)]).T
class_B_ground_truth = np.stack([np.zeros(args.batch_size), np.ones(args.batch_size)]).T


# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G(A, B, A2B=None, B2A=None, A2B2A=None, B2A2B=None):
    with tf.GradientTape() as t:
        if args.attention_type == "spa-gan" or args.attention_type == "none":
            # Transformation
            A2B = G_A2B(A, training=True)
            B2A = G_B2A(B, training=True)
            # Cycle
            A2B2A = G_B2A(A2B, training=True)
            B2A2B = G_A2B(B2A, training=True)

        # Identity
        A2A = G_B2A(A, training=True)
        B2B = G_A2B(B, training=True)

        A2B_d_logits = D_B(A2B, training=True)
        B2A_d_logits = D_A(B2A, training=True)

        A2B_g_loss = g_loss_fn(A2B_d_logits)
        B2A_g_loss = g_loss_fn(B2A_d_logits)
        A2A_id_loss = identity_loss_fn(A, A2A)
        B2B_id_loss = identity_loss_fn(B, B2B)

        A2B_counterfactual_loss = counterfactual_loss_nf(class_B_ground_truth, clf(A2B))
        B2A_counterfactual_loss = counterfactual_loss_nf(class_A_ground_truth, clf(B2A))

        A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)
        B2A2B_cycle_loss = cycle_loss_fn(B, B2A2B)

        G_loss = (A2B_g_loss + B2A_g_loss) + (A2B2A_cycle_loss + B2A2B_cycle_loss) * args.cycle_loss_weight \
                 + (A2A_id_loss + B2B_id_loss) * args.identity_loss_weight + \
                 (A2B_counterfactual_loss + B2A_counterfactual_loss) * args.counterfactual_loss_weight

    G_grad = t.gradient(G_loss, G_A2B.trainable_variables + G_B2A.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables + G_B2A.trainable_variables))

    return A2B, B2A, {'A2B_g_loss': A2B_g_loss,
                      'B2A_g_loss': B2A_g_loss,
                      'A2B2A_cycle_loss': A2B2A_cycle_loss,
                      'B2A2B_cycle_loss': B2A2B_cycle_loss,
                      'A2A_id_loss': A2A_id_loss,
                      'B2B_id_loss': B2B_id_loss,
                      'A2B_counterfactual_loss': A2B_counterfactual_loss,
                      'B2A_counterfactual_loss': B2A_counterfactual_loss}


@tf.function
def train_D(A, B, A2B, B2A):
    with tf.GradientTape() as t:
        A_d_logits = D_A(A, training=True)
        B2A_d_logits = D_A(B2A, training=True)
        B_d_logits = D_B(B, training=True)
        A2B_d_logits = D_B(A2B, training=True)

        # Get classes for accuracy
        """prediction_A_d = tf.math.argmax(tf.nn.softmax(A_d_logits), 0)
        prediction_B2A_d = tf.math.argmax(tf.nn.softmax(B2A_d_logits), 0)
        prediction_B_d = tf.math.argmax(tf.nn.softmax(B_d_logits), 0)
        prediction_A2B_d = tf.math.argmax(tf.nn.softmax(A2B_d_logits), 0)
        y_pred = [prediction_A_d, prediction_B2A_d, prediction_B_d, prediction_A2B_d]"""

        A_d_loss, B2A_d_loss = d_loss_fn(A_d_logits, B2A_d_logits)
        B_d_loss, A2B_d_loss = d_loss_fn(B_d_logits, A2B_d_logits)
        D_A_gp = gan.gradient_penalty(functools.partial(D_A, training=True), A, B2A, mode=args.gradient_penalty_mode)
        D_B_gp = gan.gradient_penalty(functools.partial(D_B, training=True), B, A2B, mode=args.gradient_penalty_mode)

        D_loss = (A_d_loss + B2A_d_loss) + (B_d_loss + A2B_d_loss) + (D_A_gp + D_B_gp) * args.gradient_penalty_weight

    D_grad = t.gradient(D_loss, D_A.trainable_variables + D_B.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D_A.trainable_variables + D_B.trainable_variables))

    return {'A_d_loss': A_d_loss + B2A_d_loss,
            'B_d_loss': B_d_loss + A2B_d_loss,
            'D_A_gp': D_A_gp,
            'D_B_gp': D_B_gp,
            }


def train_step(A, B, A_attention_image=None, B_attention_image=None):
    if args.attention_type == "attention-gan":  # Attention Gan approach
        if args.attention_gan_original:
            A2B, B2A, A2B2A, B2A2B = attention_gan_original(A, B, G_A2B, G_B2A, A_attention_image, B_attention_image)
        else:
            A2B, B2A, A2B2A, B2A2 = attention_gan_foreground(G_A2B, G_B2A, A_attention_image, B_attention_image)
        A2B, B2A, G_loss_dict = train_G(A, B, A2B, B2A, A2B2A, B2A2B)
    else:  # spa-gan or none
        A2B, B2A, G_loss_dict = train_G(A, B)

    # cannot autograph `A2B_pool`
    A2B = A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
    B2A = B2A_pool(B2A)  # because of the communication between CPU and GPU

    D_loss_dict = train_D(A, B, A2B, B2A)

    return G_loss_dict, D_loss_dict


@tf.function
def sample(A, B, A_attention_image=None, B_attention_image=None):
    if args.attention_type == "attention-gan":
        # Transform important areas
        A2B_foreground = G_A2B(A_attention_image.foreground, training=True)
        B2A_foreground = G_B2A(B_attention_image.foreground, training=True)
        # Combine new transformed foreground with background
        A2B = add_images(A2B_foreground, A_attention_image.background)
        B2A = add_images(B2A_foreground, B_attention_image.background)
    else:  # spa-gan or none
        A2B = G_A2B(A, training=False)
        B2A = G_B2A(B, training=False)
    # A2B2A = G_B2A(A2B, training=False)
    # B2A2B = G_A2B(B2A, training=False)
    return A2B, B2A


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B,
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
train_summary_writer = tf.summary.create_file_writer(py.join(TF_LOG_DIR + execution_id))

# sample
test_iter = iter(A_B_dataset_test)
sample_dir = py.join(output_dir, 'images')
py.mkdir(sample_dir)

# Create GradCAM++ object
if args.attention == "gradcam":
    gradcam = Gradcam(clf, model_modifier=ReplaceToLinear(), clone=True)
else:
    gradcam = GradcamPlusPlus(clf, model_modifier=ReplaceToLinear(), clone=True)

# main loop
with train_summary_writer.as_default():
    for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue

        # update epoch counter
        ep_cnt.assign_add(1)

        # train for an epoch
        batch_count = 0
        for A, B in tqdm.tqdm(A_B_dataset, desc='Inner Epoch Loop', total=len_dataset):
            A_attention_image = None
            B_attention_image = None
            if args.attention_type == "none":
                G_loss_dict, D_loss_dict = train_step(A, B)
            else:  # Attention
                A, A_heatmap = attention_maps.get_gradcam(A, gradcam, 0, attention_type=args.attention_type,
                                                          attention_intensity=args.attention_intensity)
                B, B_heatmap = attention_maps.get_gradcam(B, gradcam, 1, attention_type=args.attention_type,
                                                          attention_intensity=args.attention_intensity)
                if args.attention_type == "attention-gan":
                    # Attention-GAN splits fore and background and puts them together after transformation
                    A_attention_image = AttentionImage(A, A_heatmap)
                    B_attention_image = AttentionImage(B, B_heatmap)
                    G_loss_dict, D_loss_dict = train_step(A, B, A_attention_image, B_attention_image)
                else:  # Spa-gan
                    # Spa-gan puts the attention on the input image -> changes input img
                    G_loss_dict, D_loss_dict = train_step(A, B)

            # sample
            if G_optimizer.iterations.numpy() % 200 == 0 or G_optimizer.iterations.numpy() == 1:
                try:
                    A, B = next(test_iter)
                except StopIteration:  # When all elements finished
                    # Create new iterator
                    test_iter = iter(A_B_dataset_test)

                if args.attention_type == "none":
                    A2B, B2A, = sample(A, B)
                else:  # Attention
                    A_attention, A_heatmap = attention_maps.get_gradcam(A, gradcam, 0,
                                                                        attention_type=args.attention_type,
                                                                        attention_intensity=args.attention_intensity)
                    B_attention, B_heatmap = attention_maps.get_gradcam(B, gradcam, 1,
                                                                        attention_type=args.attention_type,
                                                                        attention_intensity=args.attention_intensity)
                    if args.attention_type == "attention-gan":
                        A_attention_image = AttentionImage(A, A_heatmap)
                        B_attention_image = AttentionImage(B, B_heatmap)
                        A2B, B2A, = sample(A, B, A_attention_image, B_attention_image)
                    elif args.attention_type == "spa-gan":
                        A2B, B2A, = sample(A_attention, B_attention)

                if args.attention_type != "none":  # Save with attention
                    if args.dataset == "mura":
                        imgs = [A, A_heatmap, A_attention, A2B, B, B_heatmap, B_attention, B2A]
                        save_mura_images_with_attention(imgs, clf, args.dataset, execution_id, ep_cnt, batch_count)
                    else:
                        save_images_with_attention(A, A_heatmap, A_attention, A2B, B, B_heatmap, B_attention, B2A, clf,
                                                   args.dataset, execution_id, ep_cnt, batch_count)
                else:  # Save without attention
                    if args.dataset == "mura":
                        imgs = [A, A_heatmap, A_attention, A2B, B, B_heatmap, B_attention, B2A]
                        save_mura_images(imgs, clf, args.dataset, execution_id, ep_cnt, batch_count)
                    else:
                        save_images(A, A2B, B, B2A, args.dataset, execution_id, ep_cnt, batch_count)
            batch_count += 1
        # # summary
        tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
        tl.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
        tl.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations,
                   name='learning rate')

        # save checkpoint
        if ep > 20 and ep < 25:
            checkpoint.save(ep)
