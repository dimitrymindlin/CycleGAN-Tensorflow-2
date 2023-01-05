from math import ceil, floor

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
from tf2lib_local.image import img_to_3_channel_tf_tensor


def print_kid_output(kid_value_list):
    #print(kid_value_list)
    mean = float("{0:.3f}".format(np.mean(kid_value_list) * 100))
    std = float("{0:.3f}".format(np.std(kid_value_list, dtype=np.float64) * 100))
    print("KID mean", mean)
    print("KID STD", std)
    return mean, std


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


def calc_KID_for_model_target_source_batched(translated_images, img_shape, A_dataset, B_dataset, translation_name,
                                             batch_size=20):
    # KID calculation with target and source domain. Used for object transformation since the background should stay
    # in the source domain and the foreground in the target domain.
    images_length = len(translated_images)
    kid_splits = 5
    iterations_per_kid_value = ceil(images_length / batch_size)

    if translation_name == "A2B":
        source_domain = A_dataset
        target_domain = B_dataset
    else:
        source_domain = B_dataset
        target_domain = A_dataset

    source_domain = source_domain.shuffle(len(source_domain))
    source_domain = source_domain.batch(int(batch_size / 2))
    target_domain = target_domain.shuffle(len(target_domain))
    target_domain = target_domain.batch(int(batch_size / 2))

    translated_images = img_to_3_channel_tf_tensor(translated_images)  # 3 channel images needed for KID
    img_shape = (img_shape[0], img_shape[1], 3)
    kid = KID(img_shape=img_shape)
    kid_value_list = []

    for tmp_samples_source, tmp_samples_target, batch_i in zip(source_domain, target_domain,
                                                               range(iterations_per_kid_value * kid_splits)):
        source_sample_tensor = img_to_3_channel_tf_tensor(tmp_samples_source)
        target_sample_tensor = img_to_3_channel_tf_tensor(tmp_samples_target)
        all_samples_tensor = tf.concat((target_sample_tensor, source_sample_tensor), axis=0)

        current_batch = batch_i % iterations_per_kid_value  # in current KID run
        try:
            translated_images_tmp = translated_images[current_batch * batch_size:(current_batch + 1) * batch_size]
        except IndexError:
            translated_images_tmp = translated_images[current_batch * batch_size:]

        if (batch_i + 1) % iterations_per_kid_value != 0:
            kid.update_state(all_samples_tensor[:len(translated_images_tmp)], translated_images_tmp)
        else:
            kid_value_list.append(float("{0:.3f}".format(kid.result().numpy())))
            kid.reset_state()
            if len(translated_images_tmp) > 50:
                kid.update_state(all_samples_tensor[:len(translated_images_tmp)], translated_images_tmp)

    print_kid_output(kid_value_list)


def calc_KID_for_model_batched(translated_images, img_shape, dataset, batch_size=60):
    # Standard KID calculation of translated images with target domain.
    images_length = len(translated_images)
    kid_splits = 5
    iterations_per_kid_value = ceil(images_length / batch_size)
    min_amount_for_kid_calculation = 50

    translated_images = img_to_3_channel_tf_tensor(translated_images)  # 3 channel images needed for KID
    img_shape = (img_shape[0], img_shape[1], 3)
    kid = KID(img_shape=img_shape)
    kid_value_list = []

    dataset = dataset.shuffle(len(dataset))
    # Batch after shuffling to get unique batches at each epoch.
    dataset = dataset.batch(batch_size)

    for tmp_samples, batch_i in zip(dataset, range(iterations_per_kid_value * kid_splits)):
        tmp_samples_tensor = img_to_3_channel_tf_tensor(tmp_samples)
        current_batch = batch_i % iterations_per_kid_value  # in current KID run
        try:
            translated_images_tmp = translated_images[current_batch * batch_size:(current_batch + 1) * batch_size]
        except IndexError:
            translated_images_tmp = translated_images[current_batch * batch_size:]

        if (batch_i + 1) % iterations_per_kid_value != 0:
            smaller_size = np.min((len(translated_images_tmp), len(tmp_samples_tensor)))
            if smaller_size > min_amount_for_kid_calculation:
                kid.update_state(tmp_samples_tensor[:smaller_size], translated_images_tmp[:smaller_size])
        else:
            kid_value_list.append(float("{0:.3f}".format(kid.result().numpy())))
            kid.reset_state()
            if len(tmp_samples_tensor) > min_amount_for_kid_calculation and len(
                    translated_images_tmp) > min_amount_for_kid_calculation:
                smaller_size = np.min((len(translated_images_tmp), len(tmp_samples_tensor)))
                kid.update_state(tmp_samples_tensor[:smaller_size], translated_images_tmp[:smaller_size])

    print_kid_output(kid_value_list)
