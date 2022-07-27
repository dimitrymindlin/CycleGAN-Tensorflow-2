from datetime import datetime
import time

import numpy as np
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import tqdm
import data
import module

# ==============================================================================
# =                                   param                                    =
# ==============================================================================
from attention_strategies import attention_strategies
from evaluation.kid import calc_KID_for_model
from imlib import generate_image
from imlib.image_holder import get_img_holders

py.arg('--dataset', default='horse2zebra')
py.arg('--datasets_dir', default='datasets')
py.arg('--load_size', type=int, default=286)  # load image to this size
py.arg('--crop_size', type=int, default=256)  # then crop to this size
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=200)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--adversarial_loss_mode', default='gan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--discriminator_loss_weight', type=float, default=1)
py.arg('--cycle_loss_weight', type=float, default=10)
py.arg('--counterfactual_loss_weight', type=float, default=0)
py.arg('--identity_loss_weight', type=float, default=0)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
py.arg('--attention', type=str, default="gradcam", choices=['gradcam', 'gradcam-plus-plus'])
py.arg('--clf_name', type=str, default="inception")
py.arg('--attention_type', type=str, default="attention-gan-original",
       choices=['attention-gan-foreground', 'none', 'attention-gan-original'])
py.arg('--current_attention_type', type=str, default="none")
py.arg('--generator', type=str, default="resnet", choices=['resnet', 'unet'])
py.arg('--discriminator', type=str, default="patch-gan", choices=['classic', 'patch-gan'])
py.arg('--load_checkpoint', type=str, default=None)
py.arg('--start_attention_epoch', type=int, default=2)

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
    print(f"Setting {args.load_checkpoint} as checkpoint.")
    execution_id = args.load_checkpoint
    output_dir = py.join(f'output_{args.dataset}/{execution_id}')

TF_LOG_DIR = f"logs/{args.dataset}/"

py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

# ==============================================================================
# =                                    data                                    =
# ==============================================================================

A2B_pool = data.ItemPool(args.pool_size)
B2A_pool = data.ItemPool(args.pool_size)

train_horses, train_zebras, test_horses, test_zebras, len_dataset = data.load_tfds_dataset(args.dataset,
                                                                                           args.crop_size)

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
counterfactual_loss_fn = tf.losses.MeanSquaredError()

class_A_ground_truth = np.stack([np.ones(args.batch_size), np.zeros(args.batch_size)]).T
class_B_ground_truth = np.stack([np.zeros(args.batch_size), np.ones(args.batch_size)]).T

clf = tf.keras.models.load_model(f"checkpoints/{args.clf_name}_{args.dataset}_512/model", compile=False)

G_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
D_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1)
D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1)

train_D_A_acc = tf.keras.metrics.BinaryAccuracy()
train_D_B_acc = tf.keras.metrics.BinaryAccuracy()


# ==============================================================================
# =                              helper functions                              =
# ==============================================================================

def calc_G_loss(A2B, B2A, A2B2A, B2A2B, A2A, B2B):
    # Calculate Losses
    A2B_d_logits = D_B(A2B, training=True)
    B2A_d_logits = D_A(B2A, training=True)

    if args.counterfactual_loss_weight > 0:
        A2B_counterfactual_loss = counterfactual_loss_fn(class_B_ground_truth,
                                                         clf(tf.image.resize(A2B, [512, 512])))
        B2A_counterfactual_loss = counterfactual_loss_fn(class_A_ground_truth,
                                                         clf(tf.image.resize(B2A, [512, 512])))
    else:
        A2B_counterfactual_loss = tf.zeros(())
        B2A_counterfactual_loss = tf.zeros(())

    A2B_g_loss = g_loss_fn(A2B_d_logits)
    B2A_g_loss = g_loss_fn(B2A_d_logits)
    A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)
    B2A2B_cycle_loss = cycle_loss_fn(B, B2A2B)
    A2A_id_loss = identity_loss_fn(A, A2A)
    B2B_id_loss = identity_loss_fn(B, B2B)

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

    return G_loss, G_loss_dict


# ==============================================================================
# =                                 train step                                 =
# ==============================================================================
@tf.function
def train_G_no_attentiion(A_img, B_img, G_A2B, G_B2A):
    # Generate images
    with tf.GradientTape() as t:
        A2B, B2A, A2B2A, B2A2B, A2A, B2B = attention_strategies.no_attention(A_img, B_img, G_A2B, G_B2A)
        G_loss, G_loss_dict = calc_G_loss(A2B, B2A, A2B2A, B2A2B, A2A, B2B)

    G_grad = t.gradient(G_loss, G_A2B.trainable_variables + G_B2A.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables + G_B2A.trainable_variables))
    return A2B, B2A, G_loss_dict


@tf.function
def train_G_attention_gan(A_img, B_img, A_attention, B_attention, A_background, B_background):
    # Generate images
    with tf.GradientTape() as t:
        A2B, B2A, A2B2A, B2A2B, A2A, B2B = attention_strategies.attention_gan(A_img, B_img, G_A2B, G_B2A, A_attention,
                                                                              B_attention, A_background, B_background)
        G_loss, G_loss_dict = calc_G_loss(A2B, B2A, A2B2A, B2A2B, A2A, B2B)

    G_grad = t.gradient(G_loss, G_A2B.trainable_variables + G_B2A.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables + G_B2A.trainable_variables))
    return A2B, B2A, G_loss_dict


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
    if args.current_attention_type == "none":
        A2B, B2A, G_loss_dict = train_G_no_attentiion(A_holder.img, B_holder.img, G_A2B, G_B2A)
    else:
        A2B, B2A, G_loss_dict = train_G_attention_gan(A_holder.img, B_holder.img,
                                                      A_holder.attention, B_holder.attention,
                                                      A_holder.background, B_holder.background)

    # cannot autograph `A2B_pool`
    A2B = A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
    B2A = B2A_pool(B2A)  # because of the communication between CPU and GPU

    D_loss_dict = train_D(A_holder.img, B_holder.img, A2B, B2A)
    train_D_A_acc.reset_states()
    train_D_B_acc.reset_states()

    return G_loss_dict, D_loss_dict


@tf.function
def sample_no_attention(A_img, B_img):
    A2B, B2A, A2B2A, B2A2B = attention_strategies.no_attention(A_img, B_img, G_A2B, G_B2A,
                                                               training=False)
    return A2B, B2A, A2B2A, B2A2B


@tf.function
def sample(A_img, B_img, A_attention, B_attention, A_background, B_background):
    A2B, B2A, A2B_transformed, B2A_transformed = attention_strategies.attention_gan(A_img, B_img, G_A2B, G_B2A,
                                                                                    A_attention,
                                                                                    B_attention, A_background,
                                                                                    B_background, training=False)
    return A2B, B2A, A2B_transformed, B2A_transformed


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
    print("restored checkpoint...")
    print(f"continuing with epoch {ep_cnt.numpy()}")
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join(TF_LOG_DIR + execution_id))

# sample
test_iter = iter(tf.data.Dataset.zip(((test_horses, test_zebras))))
sample_dir = py.join(output_dir, 'images')
py.mkdir(sample_dir)

# Create GradCAM object
if args.attention == "gradcam":
    gradcam = GradcamPlusPlus(clf, clone=True)
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
        for batch_count, (A, B) in enumerate(
                tqdm.tqdm(tf.data.Dataset.zip((train_horses, train_zebras)), desc='Inner Epoch Loop',
                          total=len_dataset)):
            A_holder, B_holder = get_img_holders(A, B, args.attention_type, args.attention,
                                                 gradcam=gradcam)
            # Select attention type
            if ep < args.start_attention_epoch:
                args.current_attention_type = "none"
            else:
                args.current_attention_type = args.attention_type

            G_loss_dict, D_loss_dict = train_step(A_holder, B_holder)

            # # summary
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
            tl.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations,
                       name='learning rate')

            # sample
            if ep == 0 or ep % 5 == 0:
                if G_optimizer.iterations.numpy() % 300 == 0 or G_optimizer.iterations.numpy() == 1:
                    try:
                        A, B = next(test_iter)
                    except StopIteration:  # When all elements finished
                        # Create new iterator
                        test_iter = iter(tf.data.Dataset.zip(((test_horses, test_zebras))))
                        A, B = next(test_iter)

                    A_holder, B_holder = get_img_holders(A, B, args.attention_type, args.attention,
                                                         gradcam=gradcam)

                    if args.current_attention_type == "none":
                        A2B, B2A, A2B2A, B2A2B = sample_no_attention(A_holder.img, B_holder.img)
                        generate_image(args, clf, A, B, A2B, B2A,
                                       execution_id, ep, batch_count,
                                       A_holder=A_holder,
                                       B_holder=B_holder,
                                       A2B2A=A2B2A,
                                       B2A2B=B2A2B)

                    else:
                        A2B, B2A, A2B_transformed, B2A_transformed = sample(A_holder.img, B_holder.img,
                                                                            A_holder.attention, B_holder.attention,
                                                                            A_holder.background, B_holder.background)

                        A_holder.transformed_part = A2B_transformed
                        B_holder.transformed_part = B2A_transformed

                        generate_image(args, clf, A, B, A2B, B2A,
                                       execution_id, ep, batch_count,
                                       A_holder=A_holder,
                                       B_holder=B_holder)

            batch_count += 1

        # Calculate KID after epoch and log
        if ep > 130 and ep % 5 == 0:
            kid_A2B_mean, kid_A2B_std = calc_KID_for_model(A2B_pool.items, "A2B", args.crop_size, train_horses,
                                                           train_zebras)
            kid_B2A_mean, kid_B2A_std = calc_KID_for_model(B2A_pool.items, "B2A", args.crop_size, train_horses,
                                                           train_zebras)
            tl.summary({'kid_A2B_mean': tf.Variable(kid_A2B_mean)}, step=ep, name='kid_A2B_mean')
            tl.summary({'kid_A2B_std': tf.Variable(kid_A2B_std)}, step=ep, name='kid_A2B_std')
            tl.summary({'kid_B2A_mean': tf.Variable(kid_A2B_mean)}, step=ep, name='kid_B2A_mean')
            tl.summary({'kid_B2A_std': tf.Variable(kid_A2B_mean)}, step=ep, name='kid_B2A_std')

        # save checkpoint
        if ep > 130 and ep % 5 == 0:
            checkpoint.save(ep)
