def no_attention_single(img, G, G_cycle, training):
    forward_mapping = G(img, training=training)
    # Cycle
    if G_cycle:
        cycled = G_cycle(forward_mapping, training=training)
    if training:
        # ID
        id_mapping = G_cycle(img, training)
        return forward_mapping, cycled, id_mapping
    else:
        if G_cycle:
            return forward_mapping, cycled  # used for sample phase during training
        return forward_mapping  # Used for testing counterfactuals


def no_attention_step(A_img, B_img, G_A2B, G_B2A, training=True):
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
    if training:
        A2B, A2B2A, A2A = no_attention_single(A_img, G_A2B, G_B2A, training)
        B2A, B2A2B, B2B = no_attention_single(B_img, G_B2A, G_A2B, training)
        return A2B, B2A, A2B2A, B2A2B, A2A, B2B
    else:
        A2B, A2B2A = no_attention_single(A_img, G_A2B, G_B2A, training)
        B2A, B2A2B = no_attention_single(B_img, G_B2A, G_A2B, training)
        return A2B, B2A, A2B2A, B2A2B
