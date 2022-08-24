from datetime import datetime
import time

import numpy as np
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import standard_datasets_loading
import tf2lib as tl
import tf2gan as gan
import tqdm
import module
from attention_strategies.spa_gan import spa_gan_step, spa_gan_step_fm
from imlib import generate_image
from imlib.image_holder import get_img_holders
from tf2lib.data.item_pool import ItemPool

py.arg('--dataset', default='horse2zebra')
py.arg('--datasets_dir', default='datasets')
py.arg('--load_size', type=int, default=286)  # load image to this size
py.arg('--crop_size', type=int, default=256)  # then crop to this size
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=181)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--adversarial_loss_weight', type=float, default=1)
py.arg('--cycle_loss_weight', type=float, default=10)
py.arg('--counterfactual_loss_weight', type=float, default=0)
py.arg('--feature_map_loss_weight', type=float, default=1)
py.arg('--identity_loss_weight', type=float, default=0)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
py.arg('--attention', type=str, default="clf", choices=['discriminator', 'clf'])
py.arg('--clf_name', type=str, default="inception")
py.arg('--clf_ckp_name', type=str, default="2022-06-04--00.00")  # Mura: 2022-06-04--00.05, H2Z: 2022-06-04--00.00
py.arg('--attention_intensity', type=float, default=1)
py.arg('--attention_type', type=str, default="spa-gan")
py.arg('--generator', type=str, default="resnet-attention", choices=['resnet', 'unet', "resnet-attention"])
py.arg('--discriminator', type=str, default="patch-gan", choices=['classic', 'patch-gan'])
py.arg('--load_checkpoint', type=str, default=None)
py.arg('--current_attention_type', type=str, default="none")
args = py.args()

# output_dir
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
    # For loading checkpoint
    execution_id = args.load_checkpoint
    output_dir = py.join(f'output_{args.dataset}/{execution_id}')

TF_LOG_DIR = f"logs/{args.dataset}/"

py.mkdir(output_dir)

# Correct settings for SPA-GAN
if args.feature_map_loss_weight > 0:
    args.generator = "resnet-attention"
args.current_attention_type = args.attention_type

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

# ==============================================================================
# =                                    data                                    =
# ==============================================================================

A2B_pool = ItemPool(args.pool_size)
B2A_pool = ItemPool(args.pool_size)

A_B_dataset, A_B_dataset_test, len_dataset_train = standard_datasets_loading.load_tfds_dataset(args.dataset,
                                                                                               args.crop_size)

# ==============================================================================
# =                                   models                                   =
# ==============================================================================


if args.generator == "resnet-attention":
    G_A2B = module.ResnetAttentionGenerator(input_shape=(args.crop_size, args.crop_size, 3))
    G_B2A = module.ResnetAttentionGenerator(input_shape=(args.crop_size, args.crop_size, 3))
else:
    G_A2B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))
    G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))

D_A = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, 3))
D_B = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, 3))

d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)
cycle_loss_fn = tf.losses.MeanAbsoluteError()
identity_loss_fn = tf.losses.MeanAbsoluteError()
feature_map_loss_fn = gan.get_feature_map_loss_fn()
counterfactual_loss_fn = tf.losses.MeanSquaredError()

class_A_ground_truth = np.stack([np.ones(args.batch_size), np.zeros(args.batch_size)]).T
class_B_ground_truth = np.stack([np.zeros(args.batch_size), np.ones(args.batch_size)]).T

# Create GradCAM object
gradcam = None
gradcam_D_A = None
gradcam_D_B = None
clf = None

if args.attention == "clf":
    clf = tf.keras.models.load_model(f"checkpoints/{args.clf_name}_{args.dataset}/{args.clf_ckp_name}/model",
                                     compile=False)
    # gradcam = Gradcam(clf, clone=True)
    gradcam = GradcamPlusPlus(clf, clone=True)


else:  # discriminator attention
    args.counterfactual_loss_weight = 0
    gradcam_D_A = Gradcam(D_A, model_modifier=ReplaceToLinear(), clone=True)
    gradcam_D_B = Gradcam(D_B, model_modifier=ReplaceToLinear(), clone=True)

G_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset_train, args.epoch_decay * len_dataset_train)
D_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset_train, args.epoch_decay * len_dataset_train)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1)
D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1)

train_D_A_acc = tf.keras.metrics.BinaryAccuracy()
train_D_B_acc = tf.keras.metrics.BinaryAccuracy()


# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_spa_gan_G_fm(A_img, B_img, A_attention, B_attention):
    """
    Generator with feature map loss (fm)
    Parameters
    ----------
    A_enhanced
    B_enhanced

    Returns
    -------

    """
    training = True
    with tf.GradientTape() as t:
        A2B, B2A, A2B2A, B2A2B, A_enhanced, B_enhanced, A_forward_feature_map, B_forward_feature_map, A_cycle_feature_map, B_cycle_feature_map = spa_gan_step_fm(
            A_img, B_img, G_A2B, G_B2A, A_attention, B_attention, training=True)

        GA_A2B_fm_loss = feature_map_loss_fn(A_forward_feature_map, A_cycle_feature_map)
        GB_B2A_fm_loss = feature_map_loss_fn(B_forward_feature_map, B_cycle_feature_map)

        if args.counterfactual_loss_weight > 0:
            A2B_counterfactual_loss = counterfactual_loss_fn(class_B_ground_truth,
                                                             clf(tf.image.resize(A2B, [512, 512],
                                                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)))
            B2A_counterfactual_loss = counterfactual_loss_fn(class_A_ground_truth,
                                                             clf(tf.image.resize(A2B, [512, 512],
                                                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)))

        else:
            A2B_counterfactual_loss = 0
            B2A_counterfactual_loss = 0

        A2B_d_logits = D_B(A2B, training=training)
        B2A_d_logits = D_A(B2A, training=training)
        A2B_g_loss = g_loss_fn(A2B_d_logits)
        B2A_g_loss = g_loss_fn(B2A_d_logits)
        A2B2A_cycle_loss = cycle_loss_fn(A_enhanced, A2B2A)
        B2A2B_cycle_loss = cycle_loss_fn(B_enhanced, B2A2B)

        G_loss = (A2B_g_loss + B2A_g_loss) + (A2B2A_cycle_loss + B2A2B_cycle_loss) * args.cycle_loss_weight + \
                 (GA_A2B_fm_loss + GB_B2A_fm_loss) * args.feature_map_loss_weight + \
                 (A2B_counterfactual_loss + B2A_counterfactual_loss) * args.counterfactual_loss_weight

    G_grad = t.gradient(G_loss, G_A2B.trainable_variables + G_B2A.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables + G_B2A.trainable_variables))

    return A2B, B2A, {'A2B_g_loss': A2B_g_loss,
                      'B2A_g_loss': B2A_g_loss,
                      'A2B2A_cycle_loss': A2B2A_cycle_loss,
                      'B2A2B_cycle_loss': B2A2B_cycle_loss,
                      'GA_A2B_fm_loss': GA_A2B_fm_loss,
                      'GA_B2A_fm_loss': GB_B2A_fm_loss,
                      'A2B_counterfactual_loss': A2B_counterfactual_loss,
                      'B2A_counterfactual_loss': B2A_counterfactual_loss}


@tf.function
def train_spa_gan_G(A_img, B_img, A_attention, B_attention):
    with tf.GradientTape() as t:
        A2B, B2A, A2B2A, B2A2B, A_enhanced, B_enhanced = spa_gan_step(A_img, B_img, G_A2B, G_B2A, A_attention,
                                                                      B_attention,
                                                                      training=True)

        A2B_d_logits = D_B(A2B, training=True)
        B2A_d_logits = D_A(B2A, training=True)

        if args.counterfactual_loss_weight > 0:
            A2B_counterfactual_loss = counterfactual_loss_fn(class_B_ground_truth,
                                                             clf(tf.image.resize(A2B, [512, 512])))
            B2A_counterfactual_loss = counterfactual_loss_fn(class_A_ground_truth,
                                                             clf(tf.image.resize(B2A, [512, 512])))
        else:
            A2B_counterfactual_loss = 0
            B2A_counterfactual_loss = 0

        A2B_g_loss = g_loss_fn(A2B_d_logits)
        B2A_g_loss = g_loss_fn(B2A_d_logits)
        A2B2A_cycle_loss = cycle_loss_fn(A_enhanced, A2B2A)
        B2A2B_cycle_loss = cycle_loss_fn(B_enhanced, B2A2B)

        G_loss = (A2B_g_loss + B2A_g_loss) + (A2B2A_cycle_loss + B2A2B_cycle_loss) * args.cycle_loss_weight

    G_grad = t.gradient(G_loss, G_A2B.trainable_variables + G_B2A.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables + G_B2A.trainable_variables))

    return A2B, B2A, {'A2B_g_loss': A2B_g_loss,
                      'B2A_g_loss': B2A_g_loss,
                      'A2B2A_cycle_loss': A2B2A_cycle_loss,
                      'B2A2B_cycle_loss': B2A2B_cycle_loss,
                      'A2B_counterfactual_loss': A2B_counterfactual_loss,
                      'B2A_counterfactual_loss': B2A_counterfactual_loss}


@tf.function
def train_D(A, B, A2B, B2A):
    with tf.GradientTape() as t:
        A_d_logits = D_A(A, training=True)
        B2A_d_logits = D_A(B2A, training=True)
        B_d_logits = D_B(B, training=True)
        A2B_d_logits = D_B(A2B, training=True)

        A_d_loss, B2A_d_loss = d_loss_fn(A_d_logits, B2A_d_logits)
        B_d_loss, A2B_d_loss = d_loss_fn(B_d_logits, A2B_d_logits)

        D_loss = (A_d_loss + B2A_d_loss) + (B_d_loss + A2B_d_loss)

    D_grad = t.gradient(D_loss, D_A.trainable_variables + D_B.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D_A.trainable_variables + D_B.trainable_variables))

    # Update training metric.
    train_D_A_acc.update_state(tf.ones_like(A_d_logits), A_d_logits)
    train_D_A_acc.update_state(tf.ones_like(B2A_d_logits), B2A_d_logits)
    train_D_B_acc.update_state(tf.zeros_like(B_d_logits), B_d_logits)
    train_D_B_acc.update_state(tf.zeros_like(A2B_d_logits), A2B_d_logits)

    return {'A_d_loss': A_d_loss + B2A_d_loss,
            'B_d_loss': B_d_loss + A2B_d_loss,
            'D_A_acc': train_D_A_acc.result(),
            'D_B_acc': train_D_B_acc.result()}


def train_step(A_holder, B_holder):
    if args.generator == "resnet-attention":
        A2B, B2A, G_loss_dict = train_spa_gan_G_fm(A_holder.enhanced_img, B_holder.enhanced_img, A_holder.attention,
                                                   B_holder.attention)
    else:
        A2B, B2A, G_loss_dict = train_spa_gan_G(A_holder.enhanced_img, B_holder.enhanced_img, A_holder.attention,
                                                B_holder.attention)

    # cannot autograph `A2B_pool`
    A2B = A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
    B2A = B2A_pool(B2A)  # because of the communication between CPU and GPU

    D_loss_dict = train_D(A_holder.img, B_holder.img, A2B, B2A)
    train_D_A_acc.reset_states()
    train_D_B_acc.reset_states()

    return G_loss_dict, D_loss_dict


@tf.function
def sample_spa_gan(A_img, B_img, A_attention, B_attention):
    return spa_gan_step(A_img, B_img, G_A2B, G_B2A, A_attention, B_attention, training=False)


@tf.function
def sample_spa_gan_fm(A_img, B_img, A_attention, B_attention):
    return spa_gan_step_fm(A_img, B_img, G_A2B, G_B2A, A_attention, B_attention, training=False)


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
                           py.join(output_dir, 'checkpoints'))
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
    print("restored checkpoint :)")
    print(f"continuing with epoch {ep_cnt.numpy()}")
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join(TF_LOG_DIR + execution_id))

# sample
test_iter = iter(A_B_dataset_test)
sample_dir = py.join(output_dir, 'images')
py.mkdir(sample_dir)

# main loop
with train_summary_writer.as_default():
    if args.generator == "resnet-attention":
        G_train_method = train_spa_gan_G_fm
        sample_method = sample_spa_gan_fm
    else:
        G_train_method = train_spa_gan_G
        sample_method = sample_spa_gan

    for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue

        # update epoch counter
        ep_cnt.assign_add(1)

        # train for an epoch
        batch_count = 0
        for A, B in tqdm.tqdm(A_B_dataset, desc='Inner Epoch Loop', total=len_dataset_train):
            A_holder, B_holder = get_img_holders(A, B, args.attention_type, args.attention, args.attention_intensity,
                                                 gradcam=gradcam, gradcam_D_A=gradcam_D_A, gradcam_D_B=gradcam_D_B)

            G_loss_dict, D_loss_dict = train_step(A_holder, B_holder)

            # # summary
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
            tl.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations,
                       name='learning rate')

            # sample
            if ep == 0 or ep > 80 or ep % 3 == 0:
                if G_optimizer.iterations.numpy() % 300 == 0 or G_optimizer.iterations.numpy() == 1:
                    try:
                        A, B = next(test_iter)
                    except StopIteration:  # When all elements finished
                        # Create new iterator
                        test_iter = iter(A_B_dataset_test)
                        A, B = next(test_iter)
                    # Get images
                    A_holder, B_holder = get_img_holders(A, B, args.attention_type, args.attention,
                                                         args.attention_intensity,
                                                         gradcam=gradcam, gradcam_D_A=gradcam_D_A,
                                                         gradcam_D_B=gradcam_D_B)

                    A2B, B2A, A_enhanced, B_enhanced = sample_method(A_holder.img, B_holder.img, A_holder.attention,
                                                                     B_holder.attention)
                    A_holder.transformed_part = A_enhanced
                    B_holder.transformed_part = B_enhanced
                    # Save images
                    generate_image(args, clf, A, B, A2B, B2A,
                                   execution_id, ep, batch_count,
                                   A_holder=A_holder,
                                   B_holder=B_holder)

            batch_count += 1

        # save checkpoint
        if ep > 90 and ep % 20 == 0:
            checkpoint.save(ep)
