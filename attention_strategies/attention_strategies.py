### Attention Gan Original
# From "Attention-GAN for Object Transfiguration in Wild Images"
from imlib.image_holder import add_images, multiply_images


def attention_gan_original(A, B, G_A2B, G_B2A, training=True):
    """
    Parameters
    ----------
    A AttentionImage of image A
    B AttentionImage of image B
    """
    # Transform whole image
    A2B_transformed = G_A2B(A.img, training=training)
    B2A_transformed = G_B2A(B.img, training=training)
    # Combine new transformed image with attention
    A2B_transformed_attention = multiply_images(A2B_transformed, A.foreground)
    A.transformed_part = A2B_transformed_attention
    B2A_transformed_attention = multiply_images(B2A_transformed, B.foreground)
    B.transformed_part = B2A_transformed_attention
    # Add background to new img
    A2B = add_images(A2B_transformed_attention, A.background)
    B2A = add_images(B2A_transformed_attention, B.background)
    # Cycle
    if training:
        A2B2A_transformed = G_B2A(A2B, training=training)
        B2A2B_transformed = G_A2B(B2A, training=training)
        # Combine new transformed image with attention
        A2B2A_transformed_attention = multiply_images(A2B2A_transformed, A.foreground)
        A2B2A = add_images(A2B2A_transformed_attention, A.background)
        B2A2B_transformed_attention = multiply_images(B2A2B_transformed, B.foreground)
        B2A2B = add_images(B2A2B_transformed_attention, B.background)
        return A2B, B2A, A2B2A, B2A2B
    else:
        return A2B, B2A


### Attention Gan adapted
# From "Attention-GAN for Object Transfiguration in Wild Images" but only transforming foreground

def attention_gan_foreground(A, B, G_A2B, G_B2A, training=True):
    """
    Parameters
    ----------
    A AttentionImage of image A
    B AttentionImage of image B
    """
    # Transform important areas
    A2B_foreground = G_A2B(A.foreground, training=training)
    A.transformed_part = A2B_foreground
    B2A_foreground = G_B2A(B.foreground, training=training)
    B.transformed_part = B2A_foreground
    # Combine new transformed foreground with background
    A2B = add_images(A2B_foreground, A.background)
    B2A = add_images(B2A_foreground, B.background)
    if training:
        # Cycle
        A2B2A_foreground = G_B2A(A.transformed_part, training=training)
        A2B2A = add_images(A2B2A_foreground, A.background)
        B2A2B_foreground = G_A2B(B.transformed_part, training=training)
        B2A2B = add_images(B2A2B_foreground, B.background)
        return A2B, B2A, A2B2A, B2A2B
    else:
        return A2B, B2A


def spa_gan(A, B, G_A2B, G_B2A, training=True):
    # Transform enhanced areas
    A2B = G_A2B(A.enhanced_img, training=training)
    A.transformed_part = A2B
    B2A = G_B2A(B.enhanced_img, training=training)
    B.transformed_part = B2A
    if training:
        # Cycle
        A2B2A = G_B2A(A2B, training=training)
        B2A2B = G_A2B(A2B, training=training)
        return A2B, B2A, A2B2A, B2A2B
    else:
        return A2B, B2A

def no_attention(A, B, G_A2B, G_B2A, training=True):
    A2B = G_A2B(A, training=True)
    B2A = G_B2A(B, training=True)
    if training:
        # Cycle
        A2B2A = G_B2A(A2B, training=True)
        B2A2B = G_A2B(B2A, training=True)
        return A2B, B2A, A2B2A, B2A2B
    else:
        return A2B, B2A