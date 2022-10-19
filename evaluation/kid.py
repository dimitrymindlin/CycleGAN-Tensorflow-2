import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import tqdm

from imlib import plot_any_img


class KID(keras.metrics.Metric):
    def __init__(self, img_shape, name="kid", **kwargs):
        super().__init__(name=name, **kwargs)

        # KID is estimated per batch and is averaged across batches
        self.kid_tracker = keras.metrics.Mean()

        # a pretrained InceptionV3 is used without its classification layer
        # transform the pixel values to the 0-255 range, then use the same
        # preprocessing as during pretraining
        self.encoder = keras.Sequential(
            [
                layers.InputLayer(input_shape=img_shape),
                keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=img_shape,
                    weights="imagenet",
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype=tf.float32)
        return (features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        # compute polynomial kernels using the two sets of features
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(
            generated_features, generated_features
        )
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        # estimate the squared maximum mean discrepancy using the average kernel values
        batch_size = tf.shape(real_features)[0]
        batch_size_f = tf.cast(batch_size, dtype=tf.float32)
        mean_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size))) / (
                batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = tf.reduce_sum(
            kernel_generated * (1.0 - tf.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        # update the average KID estimate
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()


def calc_KID_for_model_target_source(translated_images, translation_name, img_shape, A_dataset, B_dataset=None):
    # KID calculation with target and source domain. Used for object transformation since the background should stay
    # in the source domain and the foreground in the target domain.

    kid = KID(img_shape=img_shape)
    kid_value_list = []
    images_length = len(translated_images)
    #
    if translation_name == "A2B":
        source_domain = A_dataset
        target_domain = B_dataset
    else:
        source_domain = B_dataset
        target_domain = A_dataset

    target_sample_size = int(images_length / 2)
    all_target_samples = list(target_domain.take(target_sample_size * 5))
    source_sample_size = images_length - target_sample_size
    all_source_samples = list(source_domain.take(source_sample_size * 5))

    # Calc KID in splits of 5 different samples
    for i in range(5):
        if i == 4:
            target_samples = all_target_samples[i * target_sample_size:]
            source_samples = all_source_samples[i * source_sample_size:]
        else:
            target_samples = all_target_samples[i * target_sample_size:(i + 1) * target_sample_size]
            source_samples = all_source_samples[i * source_sample_size:(i + 1) * source_sample_size]
        target_sample_tensor = tf.squeeze(tf.convert_to_tensor(target_samples))
        source_sample_tensor = tf.squeeze(tf.convert_to_tensor(source_samples))
        all_samples_tensor = tf.concat((target_sample_tensor, source_sample_tensor), axis=0)
        kid.update_state(all_samples_tensor,
                         tf.convert_to_tensor(translated_images))
        kid_value_list.append(float("{0:.3f}".format(kid.result().numpy())))
        kid.reset_state()

    print(kid_value_list)
    mean = float("{0:.3f}".format(np.mean(kid_value_list) * 100))
    std = float("{0:.3f}".format(np.std(kid_value_list, dtype=np.float64) * 100))
    print("KID mean", mean)
    print("KID STD", std)
    return mean, std


def calc_KID_for_model(translated_images, img_shape, dataset):
    # Standard KID calculation of translated images with target domain.
    # max_samples = 100

    # Check if one channel images and if so, turn to 3 channel images.
    if img_shape[-1] == 1:
        img_shape = (img_shape[0], img_shape[1], 3)
        translated_images = tf.image.grayscale_to_rgb(tf.convert_to_tensor(translated_images))
    else:
        translated_images = tf.convert_to_tensor(translated_images)

    kid = KID(img_shape=img_shape)
    kid_value_list = []
    images_length = len(translated_images)
    all_samples_list = list(dataset.take(images_length * 5))
    # Calc KID in splits of 5 different target samples
    for i in tqdm.trange(5, desc='KID outer splits'):
        if i == 4:
            tmp_samples = all_samples_list[i * images_length:]
        else:
            tmp_samples = all_samples_list[i * images_length:(i + 1) * images_length]
            # Calc KID in splits because all samples don't fit in memory
        """for j in tqdm.trange(3, desc='KID inner splits'):
            if j == 2:
                current_samples = tmp_samples[j * max_samples:]
                current_translated_images = translated_images[j * max_samples:]
            else:
                current_samples = tmp_samples[j * max_samples:(j + 1) * max_samples]
                current_translated_images = translated_images[j * max_samples:(j + 1) * max_samples]
            if current_translated_images < 100:
                break"""
        tmp_samples_tensor = tf.squeeze(tf.image.grayscale_to_rgb(tf.convert_to_tensor(tmp_samples)))
        kid.update_state(tmp_samples_tensor, translated_images)
        kid_value_list.append(float("{0:.3f}".format(kid.result().numpy())))
        kid.reset_state()

    print(kid_value_list)
    mean = float("{0:.3f}".format(np.mean(kid_value_list) * 100))
    std = float("{0:.3f}".format(np.std(kid_value_list, dtype=np.float64) * 100))
    print("KID mean", mean)
    print("KID STD", std)
    return mean, std
