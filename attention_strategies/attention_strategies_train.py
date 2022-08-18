### Attention Gan Original
# From "Attention-GAN for Object Transfiguration in Wild Images"
from imlib.image_holder import add_images, multiply_images


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
        return A2B, B2A, A2B_transformed, B2A_transformed  # Return transformed parts to plot intermediate step


