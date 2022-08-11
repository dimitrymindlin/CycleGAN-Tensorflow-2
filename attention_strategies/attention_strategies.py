### Attention Gan Original
# From "Attention-GAN for Object Transfiguration in Wild Images"
from imlib import scale_to_minus_one_one
from imlib.image_holder import add_images, multiply_images


def no_attention(A_img, B_img, G_A2B, G_B2A, training=True):
    """
    This function doesn't use any of the attention mechanisms and translates the images as the original cycle-gan.
    Parameters
    ----------
    A_img: Image from domain A
    B_img: Image from domain B
    G_A2B: Generator from domain A to B
    G_B2A: Generator from domain B to A
    training: boolean whether training or infer

    Returns
    -------

    """
    A2B = G_A2B(A_img, training=training)
    B2A = G_B2A(B_img, training=training)
    # Cycle
    A2B2A = G_B2A(A2B, training=training)
    B2A2B = G_A2B(B2A, training=training)
    if training:
        # ID
        A2A = G_B2A(A_img, training=training)
        B2B = G_A2B(B_img, training=training)
        return A2B, B2A, A2B2A, B2A2B, A2A, B2B
    else:
        return A2B, B2A, A2B2A, B2A2B


def attention_gan(A_img, B_img, G_A2B, G_B2A, A_attention, B_attention, A_background, B_background, training=True):
    A2B_transformed = G_A2B(A_img, training=True)
    B2A_transformed = G_B2A(B_img, training=True)
    # Combine new transformed image with attention -> Crop important part from transformed img
    A2B_transformed_attention = multiply_images(A2B_transformed, A_attention)
    B2A_transformed_attention = multiply_images(B2A_transformed, B_attention)
    # Add background to new img
    A2B = add_images(A2B_transformed_attention, A_background)
    B2A = add_images(B2A_transformed_attention, B_background)
    if training:
        # Cycle
        A2B2A_transformed = G_B2A(A2B, training=True)
        B2A2B_transformed = G_A2B(B2A, training=True)
        # Combine new transformed image with attention
        A2B2A_transformed_attention = multiply_images(A2B2A_transformed, A_attention)
        A2B2A = add_images(A2B2A_transformed_attention, A_background)
        B2A2B_transformed_attention = multiply_images(B2A2B_transformed, B_attention)
        B2A2B = add_images(B2A2B_transformed_attention, B_background)
        # ID
        A2A_transformed = G_B2A(A_img, training=True)
        A2A_transformed_attention = multiply_images(A2A_transformed, A_attention)
        A2A = add_images(A2A_transformed_attention, A_background)
        B2B_transformed = G_A2B(B_img, training=True)
        B2B_transformed_attention = multiply_images(B2B_transformed, B_attention)
        B2B = add_images(B2B_transformed_attention, B_background)
        return A2B, B2A, A2B2A, B2A2B, A2A, B2B
    else:
        return A2B, B2A, A2B_transformed, B2A_transformed


def spa_gan(A, B, G_A2B, G_B2A, training=True):
    # Transform enhanced img
    A2B = G_A2B(A.enhanced_img, training=training)
    A.transformed_part = A2B
    B2A = G_B2A(B.enhanced_img, training=training)
    B.transformed_part = B2A
    if training:
        # Cycle back transformed enhanced img
        A2B2A = G_B2A(A2B, training=training)
        B2A2B = G_A2B(B2A, training=training)
        return A2B, B2A, A2B2A, B2A2B
    else:
        return A2B, B2A


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
