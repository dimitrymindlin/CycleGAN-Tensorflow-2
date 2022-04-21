### Attention Gan Original
# From "Attention-GAN for Object Transfiguration in Wild Images"
from imlib.attention_image import add_images, multiply_images


def attention_gan_original(A, B, G_A2B, G_B2A, A_attention_image, B_attention_image, training=True):
    # Transform whole image
    A2B_transformed = G_A2B(A, training=training)
    B2A_transformed = G_B2A(B, training=training)
    # Combine new transformed image with attention
    A2B_transformed_attention = multiply_images(A2B_transformed, A_attention_image.foreground)
    A_attention_image.transformed_part = A2B_transformed_attention
    B2A_transformed_attention = multiply_images(B2A_transformed, B_attention_image.foreground)
    B_attention_image.transformed_part = B2A_transformed_attention
    # Add background to new img
    A2B = add_images(A2B_transformed_attention, A_attention_image.background)
    B2A = add_images(B2A_transformed_attention, B_attention_image.background)
    # Cycle
    if training:
        A2B2A_transformed = G_B2A(A2B, training=training)
        B2A2B_transformed = G_A2B(B2A, training=training)
        # Combine new transformed image with attention
        A2B2A_transformed_attention = multiply_images(A2B2A_transformed, A_attention_image.foreground)
        A2B2A = add_images(A2B2A_transformed_attention, A_attention_image.background)
        B2A2B_transformed_attention = multiply_images(B2A2B_transformed, B_attention_image.foreground)
        B2A2B = add_images(B2A2B_transformed_attention, B_attention_image.background)
        return A2B, B2A, A2B2A, B2A2B
    else:
        return A2B, B2A


### Attention Gan adapted
# From "Attention-GAN for Object Transfiguration in Wild Images" but only transforming foreground

def attention_gan_foreground(G_A2B, G_B2A, A_attention_image, B_attention_image, training=True):
    # Transform important areas
    A2B_foreground = G_A2B(A_attention_image.foreground, training=training)
    A_attention_image.transformed_part = A2B_foreground
    B2A_foreground = G_B2A(B_attention_image.foreground, training=training)
    B_attention_image.transformed_part = B2A_foreground
    # Combine new transformed foreground with background
    A2B = add_images(A2B_foreground, A_attention_image.background)
    B2A = add_images(B2A_foreground, B_attention_image.background)
    if training:
        # Cycle
        A2B2A_foreground = G_B2A(A_attention_image.transformed_part, training=training)
        A2B2A = add_images(A2B2A_foreground, A_attention_image.background)
        B2A2B_foreground = G_A2B(B_attention_image.transformed_part, training=training)
        B2A2B = add_images(B2A2B_foreground, B_attention_image.background)
        return A2B, B2A, A2B2A, B2A2B
    else:
        return A2B, B2A
