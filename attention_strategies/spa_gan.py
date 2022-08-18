from attention_maps import apply_attention_on_img
from imlib import scale_to_minus_one_one
from imlib.image_holder import multiply_images


def spa_gan_fm(A_enhanced, B_enhanced, G_A2B, G_B2A, A_attention, B_attention):
    A2B, A_real_feature_map = G_A2B(A_enhanced, training=True)
    B2A, B_real_feature_map = G_B2A(B_enhanced, training=True)
    A2B = multiply_images(A2B, scale_to_minus_one_one(A_attention))
    B2A = multiply_images(B2A, scale_to_minus_one_one(B_attention))
    A2B2A, B_fake_feature_map = G_B2A(A2B, training=True)
    B2A2B, A_fake_feature_map = G_A2B(B2A, training=True)
    A2A, _ = G_B2A(A_enhanced, training=True)
    B2B, _ = G_A2B(B_enhanced, training=True)

    """GA_A2B_fm_loss = feature_map_loss_fn(A_real_feature_map, A_fake_feature_map)
    GA_B2A_fm_loss = feature_map_loss_fn(B_real_feature_map, B_fake_feature_map)"""

    # Todo: Complete...


def spa_gan_single(img, G, G_cycle, img_attention, training):
    img_enhanced = apply_attention_on_img(img, img_attention)
    # Transform enhanced img
    transformed = G(img_enhanced, training=training)
    if training:
        # Cycle back transformed enhanced img
        cycled = G_cycle(transformed, training=training)
        return transformed, cycled
    else:
        return transformed, img_enhanced  # return enhanced img for visualisation


def spa_gan_step(A_img, B_img, G_A2B, G_B2A, A_attention, B_attention, training=True):
    if training:
        A2B, A2B2A = spa_gan_single(A_img, G_A2B, G_B2A, A_attention, training)
        B2A, B2A2B = spa_gan_single(B_img, G_B2A, G_A2B, B_attention, training)
        return A2B, B2A, A2B2A, B2A2B
    else:  # This distinction is only due to naming ... Simplify in future
        A2B, A_enhanced = spa_gan_single(A_img, G_A2B, G_B2A, A_attention, training)
        B2A, B_enhanced = spa_gan_single(B_img, G_B2A, G_A2B, B_attention, training)
        return A2B, B2A, A_enhanced, B_enhanced
