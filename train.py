import time
from datetime import datetime

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
from attention_strategies.attention_strategies import attention_gan_original, attention_gan_foreground, spa_gan, \
    no_attention
from imlib import generate_image
from imlib.image_holder import ImageHolder

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--dataset', default='horse2zebra')
py.arg('--datasets_dir', default='datasets')
py.arg('--load_size', type=int, default=512)  # load image to this size
py.arg('--crop_size', type=int, default=512)  # then crop to this size
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=70)
py.arg('--epoch_decay', type=int, default=50)  # epoch to start decaying learning rate
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--discriminator_loss_weight', type=float, default=1)
py.arg('--cycle_loss_weight', type=float, default=1)
py.arg('--counterfactual_loss_weight', type=float, default=0)
py.arg('--feature_map_loss_weight', type=float, default=0)
py.arg('--identity_loss_weight', type=float, default=0)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
"""pool_size: the discriminator is trained against the current batch of generated images as well as images generated on 
previous iterations. Essentially, we remember the last pool_size generated images then randomly sample from this pool 
to create a batch_size batch of images to do one iteration of backprop on. This helps to stabilize training, kind of 
like experience replay."""
py.arg('--attention', type=str, default="gradcam-plus-plus", choices=['gradcam', 'gradcam-plus-plus'])
py.arg('--attention_type', type=str, default="none",
       choices=['attention-gan-foreground', 'spa-gan', 'none', 'attention-gan-original'])
py.arg('--attention_intensity', type=float, default=1)
py.arg('--generator', type=str, default="resnet", choices=['resnet', 'unet'])
py.arg('--discriminator', type=str, default="patch-gan", choices=['classic', 'patch-gan'])
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
if args.feature_map_loss_weight > 0:
    feature_map_G_A2B = tf.keras.models.Model(inputs=G_A2B.inputs, outputs=G_A2B.get_layer(name="upsampling_0").output)
    feature_map_G_B2A = tf.keras.models.Model(inputs=G_B2A.inputs, outputs=G_B2A.get_layer(name="upsampling_0").output)

if args.discriminator == "patch-gan":
    D_A = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, 3))
    D_B = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, 3))
else:
    D_A = module.ClassicDiscriminator(input_shape=(args.crop_size, args.crop_size, 3))
    D_B = module.ClassicDiscriminator(input_shape=(args.crop_size, args.crop_size, 3))

d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)
cycle_loss_fn = tf.losses.MeanAbsoluteError()
identity_loss_fn = tf.losses.MeanAbsoluteError()
counterfactual_loss_fn = tf.losses.MeanSquaredError()
feature_map_loss_fn = tf.losses.MeanAbsoluteError()

if args.dataset == "mura":
    clf = tf.keras.models.load_model(f"checkpoints/2022-03-24--12.42/model", compile=False)
else:
    clf = tf.keras.models.load_model(f"checkpoints/inception_{args.dataset}/model", compile=False)

G_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
D_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1)
D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1)
optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1)
D_A.compile(loss='mse',
            optimizer=D_optimizer,
            metrics=['accuracy'])
D_B.compile(loss='mse',
            optimizer=D_optimizer,
            metrics=['accuracy'])

patch = int(args.crop_size / 2 ** 4)
disc_patch = (patch, patch, 1)
valid = np.ones((args.batch_size,) + disc_patch)
fake = np.zeros((args.batch_size,) + disc_patch)

class_A_ground_truth = np.stack([np.ones(args.batch_size), np.zeros(args.batch_size)]).T
class_B_ground_truth = np.stack([np.zeros(args.batch_size), np.ones(args.batch_size)]).T

if args.attention_type == "spa-gan":
    attention_strategy = spa_gan
elif args.attention_type == "attention-gan-foreground":
    attention_strategy = attention_gan_foreground
elif args.attention_type == "attention-gan-original":
    attention_strategy = attention_gan_original
else:
    attention_strategy = no_attention


# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

# @tf.function
def train_G(A, B, A2B=None, B2A=None, A2B2A=None, B2A2B=None):
    with tf.GradientTape() as t:
        # cycle loss
        A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)
        B2A2B_cycle_loss = cycle_loss_fn(B, B2A2B)
        A2B_d_logits = D_B(A2B, training=True)
        B2A_d_logits = D_A(B2A, training=True)
        # adversarial loss
        A2B_g_loss = g_loss_fn(A2B_d_logits)
        B2A_g_loss = g_loss_fn(B2A_d_logits)
        # combined loss
        G_loss = (A2B_g_loss + B2A_g_loss) * args.discriminator_loss_weight \
                 + (A2B2A_cycle_loss + B2A2B_cycle_loss) * args.cycle_loss_weight
        # Identity loss
        #if args.identity_loss_weight > 0:
        A2A = G_B2A(A, training=True)
        B2B = G_A2B(B, training=True)
        A2A_id_loss = identity_loss_fn(A, A2A)
        B2B_id_loss = identity_loss_fn(B, B2B)
        G_loss += (A2A_id_loss + B2B_id_loss) * args.identity_loss_weight
        loss_dict = {'A2B_g_loss': A2B_g_loss,
                      'B2A_g_loss': B2A_g_loss,
                      'A2B2A_cycle_loss': A2B2A_cycle_loss,
                      'B2A2B_cycle_loss': B2A2B_cycle_loss,
                      'A2A_id_loss': A2A_id_loss,
                      'B2B_id_loss': B2B_id_loss}
        # counterfactual loss
        if args.counterfactual_loss_weight > 0:
            A2B_counterfactual_loss = counterfactual_loss_fn(class_B_ground_truth, clf(A2B))
            B2A_counterfactual_loss = counterfactual_loss_fn(class_A_ground_truth, clf(B2A))
            G_loss += (A2B_counterfactual_loss + B2A_counterfactual_loss) * args.counterfactual_loss_weight
            loss_dict["A2B_counterfactual_loss"] = A2B_counterfactual_loss
            loss_dict["B2A_counterfactual_loss"] = B2A_counterfactual_loss
        # feature map loss
        if args.feature_map_loss_weight > 0:
            A2B_feature_map_real = feature_map_G_A2B(A)
            A2B_feature_map_fake = feature_map_G_A2B(B2A)
            for i in range(A2B_feature_map_real.shape[3]):
                loss_sum = feature_map_loss_fn(A2B_feature_map_real[:, :, :, i], A2B_feature_map_fake[:, :, :, i])
            A2B_feature_map_loss = loss_sum / A2B_feature_map_real.shape[3]
            B2A_feature_map_real = feature_map_G_B2A(B)
            B2A_feature_map_fake = feature_map_G_B2A(A2B)
            for i in range(B2A_feature_map_real.shape[3]):
                loss_sum = feature_map_loss_fn(B2A_feature_map_real[:, :, :, i], B2A_feature_map_fake[:, :, :, i])
            B2A_feature_map_loss = loss_sum / B2A_feature_map_fake.shape[3]
            G_loss += (A2B_feature_map_loss + B2A_feature_map_loss) * args.feature_map_loss_weight
            loss_dict["A2B_feature_map_loss"] = A2B_feature_map_loss
            loss_dict["B2A_feature_map_loss"] = B2A_feature_map_loss

        # combined loss
        """G_loss = (A2B_g_loss + B2A_g_loss) * args.discriminator_loss_weight \
                 + (A2B2A_cycle_loss + B2A2B_cycle_loss) * args.cycle_loss_weight \
                 + (A2A_id_loss + B2B_id_loss) * args.identity_loss_weight + \
                 (A2B_counterfactual_loss + B2A_counterfactual_loss) * args.counterfactual_loss_weight + \
                 (A2B_feature_map_loss + B2A_feature_map_loss) * args.feature_map_loss_weight"""

    G_grad = t.gradient(G_loss, G_A2B.trainable_variables + G_B2A.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables + G_B2A.trainable_variables))

    return A2B, B2A, loss_dict


# @tf.function
def train_D(A, B, A2B, B2A):
    if args.attention_type == "attention-gan-original":
        A2B = A.transformed_part
        B2A = B.transformed_part
        A = A.foreground
        B = B.foreground
    else:
        A = A.img
        B = B.img

    with tf.GradientTape() as t:
        # Train the discriminators (original images = real (valid) / translated = Fake)
        D_A_real = D_A.train_on_batch(A, valid)
        D_A_fake = D_A.train_on_batch(B2A, fake)
        D_A_loss = 0.5 * np.add(D_A_real, D_A_fake)

        D_B_real = D_B.train_on_batch(B, valid)
        D_B_fake = D_B.train_on_batch(A2B, fake)
        D_B_loss = 0.5 * np.add(D_B_real, D_B_fake)

        # Total disciminator loss
        total_loss = 0.5 * np.add(D_A_loss, D_B_loss)

    return {'D_A_loss': D_A_loss[0],
            'D_B_loss': D_B_loss[0],
            'D_A_accuracy': D_A_loss[1],
            'D_B_accuracy': D_B_loss[1],
            'Total_loss': total_loss[0],
            'Total_accuracy': total_loss[1]
            }


def train_step(A_holder, B_holder):
    """
    Parameters
    ----------
    A ImageHolder of image A
    B ImageHolder of image B
    -------
    """
    A2B, B2A, A2B2A, B2A2B = attention_strategy(A_holder, B_holder, G_A2B, G_B2A, training=True)

    if args.attention_type == "spa-gan":
        A2B, B2A, G_loss_dict = train_G(A_holder.enhanced_img, B_holder.enhanced_img, A2B, B2A, A2B2A, B2A2B)
    else:
        A2B, B2A, G_loss_dict = train_G(A_holder.img, B_holder.img, A2B, B2A, A2B2A, B2A2B)

    # cannot autograph `A2B_pool`
    A2B = A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
    B2A = B2A_pool(B2A)  # because of the communication between CPU and GPU

    D_loss_dict = train_D(A_holder, B_holder, A2B, B2A)

    return G_loss_dict, D_loss_dict


# @tf.function
def sample(A_holder, B_holder):
    """
    Parameters
    ----------
    A AttentionImage if attention mode, else normal image tensor
    B AttentionImage if attention mode, else normal image tensor
    -------
    """
    return attention_strategy(A_holder, B_holder, G_A2B, G_B2A, training=False)


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

# Create GradCAM object
if args.attention == "gradcam":
    gradcam = Gradcam(clf, model_modifier=ReplaceToLinear(), clone=True)
elif args.attention == "gradcam-plus-plus":
    gradcam = GradcamPlusPlus(clf, model_modifier=ReplaceToLinear(), clone=True)
else:
    gradcam = None

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
            if args.attention_type == "none":
                A_holder = ImageHolder(A, 0, attention=False)
                B_holder = ImageHolder(B, 1, attention=False)
            else:  # Attention-strategies
                A_holder = ImageHolder(A, 0, gradcam, args.attention_type)
                B_holder = ImageHolder(B, 1, gradcam, args.attention_type)

            G_loss_dict, D_loss_dict = train_step(A_holder, B_holder)

            # summary
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
            tl.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations,
                       name='learning rate')

            # sample every few epochs
            if ep == 0 or ep > 15 or ep % 3 == 0:
                if G_optimizer.iterations.numpy() % 300 == 0 or G_optimizer.iterations.numpy() == 1:
                    try:
                        A, B = next(test_iter)
                    except StopIteration:  # When all elements finished
                        # Create new iterator
                        test_iter = iter(A_B_dataset_test)
                    # Get images
                    A_holder = ImageHolder(A, 0, gradcam, args.attention_type)
                    B_holder = ImageHolder(B, 1, gradcam, args.attention_type)
                    A2B, B2A = sample(A_holder, B_holder)

                    # Save images
                    generate_image(args, clf, A, B, A2B, B2A,
                                   execution_id, ep, batch_count,
                                   A_attention_image=A_holder,
                                   B_attention_image=B_holder)

            batch_count += 1

        # save checkpoint
        if ep % 5 == 0:
            checkpoint.save(ep)
