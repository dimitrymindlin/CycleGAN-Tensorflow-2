import tensorflow as tf


class CycleGAN:
    def __init__(self):
        pass

    def cycleGAN_transform(img, G, G_cycle, training):
        """
        Transforms img in target domain and returns the transformed image and the cycled image.
        Parameters
        ----------
        img : Image from source domain
        G : Generator from source to target domain
        G_cycle : Generator from target to source domain
        training : boolean whether training or inferencing

        Returns (forward_mapping, cycled, id_mapping)
        -------
        """
        forward_mapping = G(img, training=training)
        # resize to original size
        # TODO: (Dimi) is this necessary? why not use the original size?
        forward_mapping = tf.image.resize_with_pad(forward_mapping, tf.shape(img)[1], tf.shape(img)[2],
                                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Cycle
        if G_cycle:
            cycled = G_cycle(forward_mapping, training=training)
            # resize to original size
            cycled = tf.image.resize_with_pad(cycled, tf.shape(img)[1], tf.shape(img)[2],
                                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        if training:
            # ID
            id_mapping = G_cycle(img, training)
            # resize to original size
            id_mapping = tf.image.resize_with_pad(id_mapping, tf.shape(img)[1], tf.shape(img)[2],
                                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            return forward_mapping, cycled, id_mapping
        else:
            if G_cycle:
                return forward_mapping, cycled  # used for sample phase during training
            return forward_mapping  # Used for testing counterfactuals


def cycleGAN_step(A_img, B_img, G_A2B, G_B2A, training=True):
    """
    Original CycleGAN step, no attention used.
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
        A2B, A2B2A, A2A = cycleGAN_transform(A_img, G_A2B, G_B2A, training)
        B2A, B2A2B, B2B = cycleGAN_transform(B_img, G_B2A, G_A2B, training)
        return A2B, B2A, A2B2A, B2A2B, A2A, B2B
    else:
        A2B, A2B2A = cycleGAN_transform(A_img, G_A2B, G_B2A, training)
        B2A, B2A2B = cycleGAN_transform(B_img, G_B2A, G_A2B, training)
        return A2B, B2A, A2B2A, B2A2B
