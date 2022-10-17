### Attention Gan Original
# From "Attention-GAN for Object Transfiguration in Wild Images"
from imlib.image_holder import add_images, multiply_images
import tensorflow as tf

def attention_gan_image_fusion(transformed, attention, background):
    # Combine new transformed image with attention -> Crop important part from transformed img
    transformed_attention = multiply_images(transformed, attention)
    # Add background to new img
    final_transformed = add_images(transformed_attention, background)
    return final_transformed


def attention_gan_single(img, G, G_cycle, img_attention, img_background, training):
    forward_mapping = G(img, training)
    transformed = attention_gan_image_fusion(forward_mapping, img_attention, img_background)
    if training:
        # Cycle
        cycled = attention_gan_image_fusion(G_cycle(transformed, training), img_attention, img_background)
        # ID
        identity_mapping = attention_gan_image_fusion(G_cycle(img, training), img_attention, img_background)
        return transformed, cycled, identity_mapping
    else:
        return transformed, forward_mapping  # Return forward_mapping to plot intermediate step


def attention_gan_step(A_img, B_img, G_A2B, G_B2A, A_attention, B_attention, A_background, B_background, training=True):
    if training:
        A2B, A2B2A, A2A = attention_gan_single(A_img, G_A2B, G_B2A, A_attention, A_background, training)
        B2A, B2A2B, B2B = attention_gan_single(B_img, G_B2A, G_A2B, B_attention, B_background, training)
        return A2B, B2A, A2B2A, B2A2B, A2A, B2B
    else:
        A2B, A2B_forward_mapping = attention_gan_single(A_img, G_A2B, G_B2A, A_attention, A_background, training)
        B2A, B2A_forward_mapping = attention_gan_single(B_img, G_B2A, G_A2B, B_attention, B_background, training)
        return A2B, B2A, A2B_forward_mapping, B2A_forward_mapping  # Return forward_mapping to plot intermediate step


def attention_gan_discriminator_step(img, img_translated, img_attention):
    # Apply attention to img and to translated img for attentive-discriminator
    attended_img = multiply_images(img, img_attention)
    attended_translated = multiply_images(img_translated, img_attention)
    return attended_img, attended_translated
