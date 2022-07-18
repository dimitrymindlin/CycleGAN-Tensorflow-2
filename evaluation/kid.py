import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np

from imlib import plot_any_img


class KID(keras.metrics.Metric):
    def __init__(self, image_size, name="kid", **kwargs):
        super().__init__(name=name, **kwargs)

        # KID is estimated per batch and is averaged across batches
        self.kid_tracker = keras.metrics.Mean()

        # a pretrained InceptionV3 is used without its classification layer
        # transform the pixel values to the 0-255 range, then use the same
        # preprocessing as during pretraining
        self.encoder = keras.Sequential(
            [
                layers.InputLayer(input_shape=(image_size, image_size, 3)),
                keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=(image_size, image_size, 3),
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


def calc_KID_for_model(translated_images, translation_name, crop_size, A_dataset, B_dataset = None):
    kid = KID(image_size=crop_size)
    kid_value_list = []
    images_length = len(translated_images)
    if B_dataset:
        if translation_name == "A2B":
            source_domain = A_dataset
            target_domain = B_dataset
        else:
            source_domain = B_dataset
            target_domain = A_dataset

        target_sample_size = int(images_length / 2)
        all_target_samples = list(target_domain.take(target_sample_size*5))
        source_sample_size = images_length - int(images_length / 2)
        all_source_samples = list(source_domain.take(source_sample_size*5))

        for i in range(5):
            if i == 4:
                target_samples = all_target_samples[i * target_sample_size:]
                source_samples = all_source_samples[i * source_sample_size:]
            else:
                target_samples = all_target_samples[i*target_sample_size:(i+1)*target_sample_size]
                source_samples = all_source_samples[i*source_sample_size:(i+1)*source_sample_size]
            target_sample_tensor = tf.squeeze(tf.convert_to_tensor(target_samples))
            source_sample_tensor = tf.squeeze(tf.convert_to_tensor(source_samples))
            all_samples_tensor = tf.concat((target_sample_tensor, source_sample_tensor), axis=0)
            kid.update_state(all_samples_tensor,
                             tf.convert_to_tensor(translated_images))
            kid_value_list.append(float("{0:.3f}".format(kid.result().numpy())))
            kid.reset_state()
    else:
        all_samples_list = list(A_dataset.take(images_length * 5))
        for i in range(5):
            if i == 4:
                tmp_samples = all_samples_list[i * images_length:]
            else:
                tmp_samples = all_samples_list[i*images_length:(i+1)*images_length]

            tmp_samples_tensor = tf.squeeze(tf.convert_to_tensor(tmp_samples))
            kid.update_state(tmp_samples_tensor,
                             tf.convert_to_tensor(translated_images))
            kid_value_list.append(float("{0:.3f}".format(kid.result().numpy())))
            kid.reset_state()

    print(kid_value_list)
    mean = float("{0:.3f}".format(np.mean(kid_value_list) * 100))
    std = float("{0:.3f}".format(np.std(kid_value_list, dtype=np.float64) * 100))
    print("KID mean", mean)
    print("KID STD", std)
