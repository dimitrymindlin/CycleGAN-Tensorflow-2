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