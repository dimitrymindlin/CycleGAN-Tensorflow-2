from datetime import datetime, time

from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import tqdm
from imlib import generate_image
import data
import module
import numpy as np
# ==============================================================================
# =                                   param                                    =
# ==============================================================================
from imlib.image_holder import get_img_holders, multiply_images, add_images

py.arg('--dataset', default='horse2zebra')
py.arg('--datasets_dir', default='datasets')
py.arg('--load_size', type=int, default=530)  # load image to this size
py.arg('--crop_size', type=int, default=512)  # then crop to this size
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
py.arg('--attention', type=str, default="gradcam-plus-plus", choices=['gradcam', 'gradcam-plus-plus'])
py.arg('--attention_type', type=str, default="attention-gan-original",
       choices=['attention-gan-foreground', 'none', 'attention-gan-original'])
py.arg('--generator', type=str, default="resnet", choices=['resnet', 'unet'])
py.arg('--discriminator', type=str, default="patch-gan", choices=['classic', 'patch-gan'])
py.arg('--load_checkpoint', type=str, default=None)
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

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

# ==============================================================================
# =                                    data                                    =
# ==============================================================================

A_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, 'trainA'), '*.jpg')
B_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, 'trainB'), '*.jpg')
A_B_dataset, len_dataset = data.make_zip_dataset(A_img_paths, B_img_paths, args.batch_size, args.load_size,
                                                 args.crop_size, training=True, repeat=False)

A2B_pool = data.ItemPool(args.pool_size)
B2A_pool = data.ItemPool(args.pool_size)

A_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testA'), '*.jpg')
B_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testB'), '*.jpg')
A_B_dataset_test, _ = data.make_zip_dataset(A_img_paths_test, B_img_paths_test, args.batch_size, args.load_size,
                                            args.crop_size, training=False, repeat=True)

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

clf = tf.keras.models.load_model(f"checkpoints/inception_{args.dataset}/model", compile=False)

G_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
D_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1)
D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1)

train_D_A_acc = tf.keras.metrics.BinaryAccuracy()
train_D_B_acc = tf.keras.metrics.BinaryAccuracy()


# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G_attention_gan(A_img, B_img, A_attention, B_attention, A_background, B_background):
    with tf.GradientTape() as t:
        A2B_transformed = G_A2B(A_img, training=True)
        B2A_transformed = G_B2A(B_img, training=True)
        # Combine new transformed image with attention -> Crop important part from transformed img
        A2B_transformed_attention = multiply_images(A2B_transformed, A_attention)
        B2A_transformed_attention = multiply_images(B2A_transformed, B_attention)
        # Add background to new img
        A2B = add_images(A2B_transformed_attention, A_background)
        B2A = add_images(B2A_transformed_attention, B_background)
        # Cycle
        A2B2A_transformed = G_B2A(A2B, training=True)
        B2A2B_transformed = G_A2B(B2A, training=True)
        # Combine new transformed image with attention
        A2B2A_transformed_attention = multiply_images(A2B2A_transformed, A_attention)
        A2B2A = add_images(A2B2A_transformed_attention, A_background)
        B2A2B_transformed_attention = multiply_images(B2A2B_transformed, B_attention)
        B2A2B = add_images(B2A2B_transformed_attention, B_background)

        A2A_transformed = G_B2A(A_img, training=True)
        A2A_transformed_attention = multiply_images(A2A_transformed, A_attention)
        A2A = add_images(A2A_transformed_attention, A_background)
        B2B_transformed = G_A2B(B_img, training=True)
        B2B_transformed_attention = multiply_images(B2B_transformed, B_attention)
        B2B = add_images(B2B_transformed_attention, B_background)

        A2B_d_logits = D_B(A2B, training=True)
        B2A_d_logits = D_A(B2A, training=True)

        A2B_g_loss = g_loss_fn(A2B_d_logits)
        B2A_g_loss = g_loss_fn(B2A_d_logits)
        A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)
        B2A2B_cycle_loss = cycle_loss_fn(B, B2A2B)
        A2A_id_loss = identity_loss_fn(A, A2A)
        B2B_id_loss = identity_loss_fn(B, B2B)

        G_loss = (A2B_g_loss + B2A_g_loss) + (A2B2A_cycle_loss + B2A2B_cycle_loss) * args.cycle_loss_weight + (
                A2A_id_loss + B2B_id_loss) * args.identity_loss_weight

    G_grad = t.gradient(G_loss, G_A2B.trainable_variables + G_B2A.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables + G_B2A.trainable_variables))

    return A2B, B2A, {'A2B_g_loss': A2B_g_loss,
                      'B2A_g_loss': B2A_g_loss,
                      'A2B2A_cycle_loss': A2B2A_cycle_loss,
                      'B2A2B_cycle_loss': B2A2B_cycle_loss,
                      'A2A_id_loss': A2A_id_loss,
                      'B2B_id_loss': B2B_id_loss}


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
def sample(A_img, B_img,
           A_attention, B_attention,
           A_background, B_background):
    A2B_transformed = G_A2B(A_img, training=False)
    B2A_transformed = G_B2A(B_img, training=False)
    # Combine new transformed image with attention -> Crop important part from transformed img
    A2B_transformed_attention = multiply_images(A2B_transformed, A_attention)
    B2A_transformed_attention = multiply_images(B2A_transformed, B_attention)
    # Add background to new img
    A2B = add_images(A2B_transformed_attention, A_background)
    B2A = add_images(B2A_transformed_attention, B_background)
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
                           py.join(output_dir, 'checkpoints'),
                           max_to_keep=5)
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

# Create GradCAM object
if args.attention == "gradcam-plus-plus":
    gradcam = GradcamPlusPlus(clf, model_modifier=ReplaceToLinear(), clone=True)
else:
    gradcam = None

# main loop
with train_summary_writer.as_default():
    for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            print(f"skipping epoch {ep}")
            continue

        # update epoch counter
        ep_cnt.assign_add(1)

        # train for an epoch
        for batch_count, (A, B) in enumerate(tqdm.tqdm(A_B_dataset, desc='Inner Epoch Loop', total=len_dataset)):
            A_holder, B_holder = get_img_holders(A, B, args.attention_type, args.attention,
                                                 gradcam=gradcam)

            G_loss_dict, D_loss_dict = train_step(A_holder, B_holder)

            # # summary
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
            tl.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations,
                       name='learning rate')

            # sample
            if ep == 0 or ep > 15 or ep % 3 == 0:
                #if G_optimizer.iterations.numpy() % 300 == 0 or G_optimizer.iterations.numpy() == 1:
                try:
                    A, B = next(test_iter)
                except StopIteration:  # When all elements finished
                    # Create new iterator
                    test_iter = iter(A_B_dataset_test)
                    A, B = next(test_iter)
                A_holder, B_holder = get_img_holders(A, B, args.attention_type, args.attention,
                                                     gradcam=gradcam)

                A2B, B2A, A2B_transformed, B2A_transformed = sample(A_holder.img, B_holder.img,
                                                                    A_holder.attention, B_holder.attention,
                                                                    A_holder.background, B_holder.background)

                if tf.equal(tf.size(A2B), 0):
                    print("Size is zero in Before generate image")
                A_holder.transformed_part = A2B_transformed
                B_holder.transformed_part = B2A_transformed

                # Save images
                generate_image(args, clf, A, B, A2B, B2A,
                               execution_id, ep, batch_count,
                               A_holder=A_holder,
                               B_holder=B_holder)

            batch_count += 1

        # save checkpoint
        if ep % 5 == 0:
            checkpoint.save(ep)
