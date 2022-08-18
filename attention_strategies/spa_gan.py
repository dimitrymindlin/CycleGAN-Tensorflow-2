from attention_maps import apply_attention_on_img


def spa_gan_single(img, G, G_cycle, img_attention, training):
    img_enhanced = apply_attention_on_img(img, img_attention)
    # Transform enhanced img
    transformed = G(img_enhanced, training=training)
    if training:
        transformed_enhanced = apply_attention_on_img(transformed, img_attention)
        # Cycle back transformed enhanced img
        cycled = G_cycle(transformed_enhanced, training=training)
        return transformed, cycled, img_enhanced
    else:
        return transformed, img_enhanced  # return enhanced img for visualisation


def spa_gan_step(A_img, B_img, G_A2B, G_B2A, A_attention, B_attention, training=True):
    if training:
        A2B, A2B2A, A_enhanced = spa_gan_single(A_img, G_A2B, G_B2A, A_attention, training)
        B2A, B2A2B, B_enhanced = spa_gan_single(B_img, G_B2A, G_A2B, B_attention, training)
        return A2B, B2A, A2B2A, B2A2B, A_enhanced, B_enhanced
    else:
        A2B, A_enhanced = spa_gan_single(A_img, G_A2B, G_B2A, A_attention, training)
        B2A, B_enhanced = spa_gan_single(B_img, G_B2A, G_A2B, B_attention, training)
        return A2B, B2A, A_enhanced, B_enhanced


def spa_gan_single_fm(img, G, G_cycle, img_attention, training):
    img_enhanced = apply_attention_on_img(img, img_attention)
    # Transform enhanced img
    transformed, forward_feature_map = G(img_enhanced, training=training)
    if training:
        transformed_enhanced = apply_attention_on_img(transformed, img_attention)
        # Cycle back transformed enhanced img
        cycled, cycle_feature_map = G_cycle(transformed_enhanced, training=training)
        return transformed, cycled, img_enhanced, forward_feature_map, cycle_feature_map
    else:
        return transformed, img_enhanced  # return enhanced img for visualisation


def spa_gan_step_fm(A_img, B_img, G_A2B, G_B2A, A_attention, B_attention, training=True):
    if training:
        A2B, A2B2A, A_enhanced, A_forward_feature_map, A_cycle_feature_map = spa_gan_single_fm(A_img, G_A2B, G_B2A,
                                                                                               A_attention, training)
        B2A, B2A2B, B_enhanced, B_forward_feature_map, B_cycle_feature_map = spa_gan_single_fm(B_img, G_B2A, G_A2B,
                                                                                               B_attention, training)
        return A2B, B2A, A2B2A, B2A2B, A_enhanced, B_enhanced, A_forward_feature_map, B_forward_feature_map, A_cycle_feature_map, B_cycle_feature_map
    else:
        A2B, A_enhanced = spa_gan_single_fm(A_img, G_A2B, G_B2A, A_attention, training)
        B2A, B_enhanced = spa_gan_single_fm(B_img, G_B2A, G_A2B, B_attention, training)
        return A2B, B2A, A_enhanced, B_enhanced
