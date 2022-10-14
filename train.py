import os

import numpy as np
from rsna import get_rsna_ds_split_class
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
from mura import get_mura_ds_by_body_part_split_class
import standard_datasets_loading
import tf2lib as tl
import tf2gan as gan
import tqdm
import module
from attention_strategies.attention_gan import attention_gan_step, attention_gan_discriminator_step
from attention_strategies.no_attention import no_attention_step
from imlib import generate_image
from imlib.image_holder import get_img_holders
from tf2lib.data.item_pool import ItemPool

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root


def run_training(args, TFDS_PATH, TF_LOG_DIR, output_dir, execution_id):
    """try:
        # Correct necessary settings for SPA-GAN
        if args.feature_map_loss_weight > 0:
            args.generator = "resnet-attention"
            args.current_attention_type = args.attention_type
        if args.attention_type == "spa-gan":
            feature_map_loss_fn = gan.get_feature_map_loss_fn()
            gradcam = None
            gradcam_D_A = None
            gradcam_D_B = None
    except AttributeError:
        pass  # Spa-GAN not implemented yet"""

    # ==============================================================================
    # =                                    data                                    =
    # ==============================================================================
    args.img_shape = (args.crop_size, args.crop_size, args.img_channels)
    A2B_pool = ItemPool(args.pool_size)
    B2A_pool = ItemPool(args.pool_size)

    special_normalisation = tf.keras.applications.inception_v3.preprocess_input

    if args.dataset == "mura":
        # A = Normal, B = Abnormal
        A_B_dataset, A_B_dataset_valid, A_B_dataset_test, len_dataset_train = get_mura_ds_by_body_part_split_class(
            args.body_parts,
            TFDS_PATH,
            args.batch_size,
            args.crop_size,
            args.load_size,
            special_normalisation)
    elif args.dataset == "rsna":
        A_B_dataset, A_B_dataset_valid, A_B_dataset_test, len_dataset_train = get_rsna_ds_split_class(TFDS_PATH,
                                                                                                      args.batch_size,
                                                                                                      args.crop_size,
                                                                                                      args.load_size,
                                                                                                      special_normalisation)
    else:  # Load Horse2Zebra / Apple2Orange
        A_B_dataset, A_B_dataset_test, len_dataset_train = standard_datasets_loading.load_tfds_dataset(args.dataset,
                                                                                                       args.crop_size)

    # ==============================================================================
    # =                                   models                                   =
    # ==============================================================================

    if args.generator == "resnet-attention":
        G_A2B = module.ResnetAttentionGenerator(input_shape=args.img_shape)
        G_B2A = module.ResnetAttentionGenerator(input_shape=args.img_shape)
    else:
        G_A2B = module.ResnetGenerator(input_shape=args.img_shape)
        G_B2A = module.ResnetGenerator(input_shape=args.img_shape)

    D_A = module.ConvDiscriminator(input_shape=args.img_shape, norm=args.disc_norm)
    D_B = module.ConvDiscriminator(input_shape=args.img_shape, norm=args.disc_norm)

    # Losses
    d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)
    cycle_loss_fn = tf.losses.MeanAbsoluteError()
    identity_loss_fn = tf.losses.MeanAbsoluteError()
    counterfactual_loss_fn = tf.losses.MeanSquaredError()

    # Ground truth for counterfactual loss
    class_A_ground_truth = np.stack([np.ones(args.batch_size), np.zeros(args.batch_size)]).T
    class_B_ground_truth = np.stack([np.zeros(args.batch_size), np.ones(args.batch_size)]).T

    # Optimizers
    G_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset_train, args.epoch_decay * len_dataset_train)
    D_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset_train, args.epoch_decay * len_dataset_train)
    G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1)
    D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1)

    train_D_A_acc = tf.keras.metrics.BinaryAccuracy()
    train_D_B_acc = tf.keras.metrics.BinaryAccuracy()

    if args.attention_type == "attention-gan-original":
        clf = tf.keras.models.load_model(
            f"{ROOT_DIR}/checkpoints/{args.clf_name}_{args.dataset}/{args.clf_ckp_name}/model",
            compile=False)
        gradcam = GradcamPlusPlus(clf, clone=True)
    elif args.attention_type == "spa-gan":
        if args.attention == "clf":
            clf = tf.keras.models.load_model(
                f"{ROOT_DIR}/checkpoints/{args.clf_name}_{args.dataset}/{args.clf_ckp_name}/model",
                compile=False)
            gradcam = GradcamPlusPlus(clf, clone=True)
        else:  # discriminator attention
            args.counterfactual_loss_weight = 0
            gradcam_D_A = Gradcam(D_A, model_modifier=ReplaceToLinear(), clone=True)
            gradcam_D_B = Gradcam(D_B, model_modifier=ReplaceToLinear(), clone=True)
            # ... Implement SPA-GAN completely?
    else:
        clf = None

    # ==============================================================================
    # =                                 train step                                 =
    # ==============================================================================
    @tf.function
    def train_G_no_attention(A_img, B_img):
        training = True
        with tf.GradientTape() as t:
            A2B, B2A, A2B2A, B2A2B, A2A, B2B = no_attention_step(A_img, B_img, G_A2B, G_B2A)
            # Calculate Losses
            A2B_d_logits = D_B(A2B, training=training)
            B2A_d_logits = D_A(B2A, training=training)

            if args.counterfactual_loss_weight > 0:
                A2B_counterfactual_loss = counterfactual_loss_fn(class_B_ground_truth,
                                                                 clf(tf.image.resize(A2B, [512, 512],
                                                                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)))
                B2A_counterfactual_loss = counterfactual_loss_fn(class_A_ground_truth,
                                                                 clf(tf.image.resize(A2B, [512, 512],
                                                                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)))

            else:
                A2B_counterfactual_loss = tf.zeros(())
                B2A_counterfactual_loss = tf.zeros(())

            A2B_g_loss = g_loss_fn(A2B_d_logits)
            B2A_g_loss = g_loss_fn(B2A_d_logits)
            A2B2A_cycle_loss = cycle_loss_fn(A_img, A2B2A)
            B2A2B_cycle_loss = cycle_loss_fn(B_img, B2A2B)
            A2A_id_loss = identity_loss_fn(A_img, A2A)
            B2B_id_loss = identity_loss_fn(B_img, B2B)

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
    def train_G_attention_gan(A_img, B_img, A_attention, B_attention, A_background, B_background):
        training = True
        with tf.GradientTape() as t:
            # Generate Images based on attention-gan strategy
            A2B, B2A, A2B2A, B2A2B, A2A, B2B = attention_gan_step(A_img, B_img, G_A2B, G_B2A, A_attention, B_attention,
                                                                  A_background, B_background, training)
            # Calculate Losses
            A2B_d_logits = D_B(A2B, training=training)
            B2A_d_logits = D_A(B2A, training=training)

            if args.counterfactual_loss_weight > 0:
                A2B_counterfactual_loss = counterfactual_loss_fn(class_B_ground_truth,
                                                                 clf(tf.image.resize(A2B, [512, 512],
                                                                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)))
                B2A_counterfactual_loss = counterfactual_loss_fn(class_A_ground_truth,
                                                                 clf(tf.image.resize(B2A, [512, 512],
                                                                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)))
            else:
                A2B_counterfactual_loss = tf.zeros(())
                B2A_counterfactual_loss = tf.zeros(())

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
        training = True
        with tf.GradientTape() as t:
            A_d_logits = D_A(A, training=training)
            B2A_d_logits = D_A(B2A, training=training)
            B_d_logits = D_B(B, training=training)
            A2B_d_logits = D_B(A2B, training=training)

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
            A2B, B2A, G_loss_dict = train_G_no_attention(A_holder.img, B_holder.img)
        else:
            A2B, B2A, G_loss_dict = train_G_attention_gan(A_holder.img, B_holder.img,
                                                          A_holder.attention, B_holder.attention,
                                                          A_holder.background, B_holder.background)

        # cannot autograph `A2B_pool`
        A2B = A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
        B2A = B2A_pool(B2A)  # because of the communication between CPU and GPU

        if args.discriminator == "patch_gan_attention" and args.current_attention_type == "attention-gan-original":
            A, A2B = attention_gan_discriminator_step(A_holder.img, A2B, A_holder.attention)
            B, B2A = attention_gan_discriminator_step(B_holder.img, B2A, B_holder.attention)
        else:
            A = A_holder.img
            B = B_holder.img
        D_loss_dict = train_D(A, B, A2B, B2A)
        train_D_A_acc.reset_states()
        train_D_B_acc.reset_states()

        return G_loss_dict, D_loss_dict

    @tf.function
    def sample_no_attention(A_img, B_img):
        training = False
        A2B, B2A, A2B2A, B2A2B = no_attention_step(A_img, B_img, G_A2B, G_B2A,
                                                   training=training)
        return A2B, B2A, A2B2A, B2A2B

    @tf.function
    def sample_attention_gan(A_img, B_img, A_attention, B_attention, A_background, B_background):
        training = False
        # Generate Images based on attention-gan strategy
        A2B, B2A, A2B_transformed, B2A_transformed = attention_gan_step(A_img, B_img, G_A2B, G_B2A, A_attention,
                                                                        B_attention, A_background, B_background,
                                                                        training)
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
    test_iter = iter(A_B_dataset_test)
    sample_dir = py.join(output_dir, 'images')
    py.mkdir(sample_dir)

    # main loop
    with train_summary_writer.as_default():
        for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
            if ep < ep_cnt:
                continue

            # update epoch counter
            ep_cnt.assign_add(1)

            # train for an epoch
            for batch_count, (A, B) in enumerate(
                    tqdm.tqdm(A_B_dataset, desc='Inner Epoch Loop', total=len_dataset_train)):
                # Select attention type
                if ep < args.start_attention_epoch:
                    args.current_attention_type = "none"
                else:
                    args.current_attention_type = args.attention_type

                A_holder, B_holder = get_img_holders(A, B, args.current_attention_type, args.attention,
                                                     gradcam=gradcam)

                G_loss_dict, D_loss_dict = train_step(A_holder, B_holder)

                # # summary
                tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
                tl.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
                tl.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations,
                           name='learning rate')

                # sample
                if ep == 0 or ep % args.sample_interval == 0:
                    if G_optimizer.iterations.numpy() % 1000 == 0 or G_optimizer.iterations.numpy() == 1:
                        try:
                            A, B = next(test_iter)
                        except StopIteration:  # When all elements finished
                            # Create new iterator
                            test_iter = iter(A_B_dataset_test)
                            A, B = next(test_iter)

                        A_holder, B_holder = get_img_holders(A, B, args.current_attention_type, args.attention,
                                                             gradcam=gradcam)

                        if args.current_attention_type == "none":
                            A2B, B2A, A2B2A, B2A2B = sample_no_attention(A_holder.img, B_holder.img)
                        else:
                            A2B, B2A, A2B_transformed, B2A_transformed = sample_attention_gan(A_holder.img,
                                                                                              B_holder.img,
                                                                                              A_holder.attention,
                                                                                              B_holder.attention,
                                                                                              A_holder.background,
                                                                                              B_holder.background)
                            A_holder.transformed_part = A2B_transformed
                            B_holder.transformed_part = B2A_transformed
                            A2B2A = None
                            B2A2B = None

                        generate_image(args, clf, A, B, A2B, B2A,
                                       execution_id, ep, batch_count,
                                       A_holder=A_holder,
                                       B_holder=B_holder,
                                       A2B2A=A2B2A,
                                       B2A2B=B2A2B)

            if (ep > (args.epochs / 2) and ep % args.sample_interval == 0) or ep == (args.epochs - 1):
                checkpoint.save(ep)
                """kid_A2B_mean, kid_A2B_std = calc_KID_for_model(A2B_pool.items, "A2B", args.img_shape, train_horses,
                                                               train_zebras)
                kid_B2A_mean, kid_B2A_std = calc_KID_for_model(B2A_pool.items, "B2A", args.img_shape, train_horses,
                                                               train_zebras)
                tl.summary({'kid_A2B_mean': tf.Variable(kid_A2B_mean)}, step=kid_count, name='kid_A2B_mean')
                tl.summary({'kid_A2B_std': tf.Variable(kid_A2B_std)}, step=kid_count, name='kid_A2B_std')
                tl.summary({'kid_B2A_mean': tf.Variable(kid_A2B_mean)}, step=kid_count, name='kid_B2A_mean')
                tl.summary({'kid_B2A_std': tf.Variable(kid_A2B_mean)}, step=kid_count, name='kid_B2A_std')
    
                kid_count += 1"""
