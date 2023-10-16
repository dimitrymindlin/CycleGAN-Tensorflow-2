import os
from pathlib import Path
from typing import List
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tf2lib as tl


def filenames(data_root: str,
              body_parts: List[str],
              train=True,
              transformed=False):
    partition_name = "train" if train else "valid"
    csv_path = os.path.join(data_root, f"MURA-v1.1/{partition_name}_image_paths.csv")

    with open(csv_path, 'rb') as F:
        d = F.readlines()
        if not transformed:
            if body_parts == 'all':
                imgs = [os.path.join(data_root, str(x, encoding='utf-8').strip()) for x in d]
            else:
                imgs = [os.path.join(data_root, str(x, encoding='utf-8').strip()) for x
                        in d if str(x, encoding='utf-8').strip().split('/')[2] in body_parts]
        else:
            imgs = [
                os.path.join(data_root, str(x, encoding='utf-8').strip().replace("MURA-v1.1", "MURA-v1.1_transformed"))
                for x in d if str(x, encoding='utf-8').strip().split('/')[2] in body_parts]
    if len(imgs) == 0:
        raise FileNotFoundError(f"Couldn't filter dataset based on {body_parts}. Check if spelling is correct.")
    labels = [x.split('_')[-1].split('/')[0] for x in imgs]
    return imgs, labels


def get_mura_data_paths(body_parts: List[str],
                        data_root: str,
                        valid_percentage=0.2,
                        transformed=False):
    """
    body_parts: List of body parts to work with. Check MURA documentation for available body_parts.
    data_root: Path to dataset root folder.
    valid_percentage: Percentage of data to use for validation.
    transformed: Whether to use transformed data or not.
    """

    # To get the filenames for a task
    def to_categorical(x, y):
        y = [0 if x == 'negative' else 1 for x in y]
        y = tf.keras.utils.to_categorical(y)
        x, y = shuffle(x, y)
        return x, y

    train_x, train_y = filenames(data_root, body_parts, transformed=transformed)  # train data
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y,
                                                          test_size=valid_percentage,
                                                          random_state=42)  # split train and valid data
    test_x, test_y = filenames(data_root, body_parts, train=False, transformed=transformed)  # test data

    train_x, train_y = to_categorical(train_x, train_y)
    valid_x, valid_y = to_categorical(valid_x, valid_y)
    test_x, test_y = to_categorical(test_x, test_y)

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def normalize_img(img, special_normalisation):
    if not special_normalisation or special_normalisation == tf.keras.applications.inception_v3.preprocess_input:
        return img / tf.reduce_max(img) * 2 - 1
    elif special_normalisation == tf.keras.applications.densenet.preprocess_input:
        return img / tf.reduce_max(img)


def make_dataset(img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=True, repeat=1,
                 labels=None, special_normalisation=None):
    """
    Returns a preprocesed batched dataset. If train=True then augmentations are applied.
    """
    if training:
        @tf.function
        def _map_fn(img, label=None):  # preprocessing
            img = tf.cast(img, tf.float32)
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_contrast(img, 0.7, 1.3)
            img = tf.image.random_brightness(img, 0.2)
            """gamma = tf.random.uniform(minval=0.8, maxval=1.2, shape=[1, ])
            img = tf.image.adjust_gamma(img, gamma=gamma[0])"""
            img = tf.image.resize_with_pad(img, load_size, load_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])
            img = normalize_img(img, special_normalisation)
            if label is not None:
                return img, label
            return img
    else:
        @tf.function
        def _map_fn(img, label=None):  # preprocessing
            img = tf.cast(img, tf.float32)
            img = tf.image.resize_with_pad(img, crop_size, crop_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])
            img = normalize_img(img, special_normalisation)
            if label is not None:
                return img, label
            return img

    return tl.disk_image_batch_dataset(img_paths,
                                       batch_size,
                                       drop_remainder=drop_remainder,
                                       map_fn=_map_fn,
                                       shuffle=shuffle,
                                       repeat=repeat,
                                       labels=labels)


def _set_repeat(repeat, A_img_paths, B_img_paths):
    # zip two datasets aligned by the longer one
    if repeat:
        A_repeat = B_repeat = None  # cycle both
    else:
        if len(A_img_paths) >= len(B_img_paths):
            A_repeat = 1
            B_repeat = None  # cycle the shorter one
        else:
            A_repeat = None  # cycle the shorter one
            B_repeat = 1
    return A_repeat, B_repeat


def make_zip_dataset(A_img_paths, B_img_paths, batch_size, load_size, crop_size, training, shuffle=False, repeat=False,
                     special_normalisation=None):
    # zip two datasets aligned by the longer one
    A_repeat, B_repeat = _set_repeat(repeat, A_img_paths, B_img_paths)

    A_dataset = make_dataset(A_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True,
                             shuffle=shuffle, repeat=A_repeat, special_normalisation=special_normalisation)
    B_dataset = make_dataset(B_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True,
                             shuffle=shuffle, repeat=B_repeat, special_normalisation=special_normalisation)

    A_B_dataset = tf.data.Dataset.zip((A_dataset, B_dataset))
    len_dataset = max(len(A_img_paths), len(B_img_paths)) // batch_size
    return A_B_dataset, len_dataset


def make_concat_dataset(A_img_paths, B_img_paths, batch_size, load_size, crop_size, training, shuffle=False,
                        special_normalisation=None):
    dataset_length = len(A_img_paths) + len(B_img_paths)
    class_labels = [(1, 0) for _ in range(len(A_img_paths))]
    class_labels.extend([(0, 1) for _ in range(len(B_img_paths))])
    A_img_paths.extend(B_img_paths)  # becoming all_image_paths
    all_image_paths = A_img_paths
    return make_dataset(all_image_paths, batch_size, load_size, crop_size, training, drop_remainder=True,
                        shuffle=shuffle, repeat=1, labels=class_labels,
                        special_normalisation=special_normalisation), dataset_length


def get_split_dataset_paths(body_parts: List[str], tfds_path: str, transformed: bool):
    """
    body_parts: List of body parts to work with. Check MURA documentation for available body_parts.
    tfds_path: Path to tensorflow datasets directory.
    """
    # A = 0 = negative, B = 1 = positive
    train_x, train_y, valid_x, valid_y, test_x, test_y = get_mura_data_paths(body_parts, tfds_path,
                                                                             transformed=transformed)
    A_img_paths = [filename for filename in train_x if "negative" in filename]
    B_img_paths = [filename for filename in train_x if "positive" in filename]
    A_img_paths_valid = [filename for filename in valid_x if "negative" in filename]
    B_img_paths_valid = [filename for filename in valid_x if "positive" in filename]
    A_img_paths_test = [filename for filename in test_x if "negative" in filename]
    B_img_paths_test = [filename for filename in test_x if "positive" in filename]
    return A_img_paths, B_img_paths, \
        A_img_paths_valid, B_img_paths_valid, \
        A_img_paths_test, B_img_paths_test


def get_mura_ds_by_body_part_split_class(body_parts, tfds_path, batch_size, crop_size, load_size,
                                         special_normalisation=None, transformed=True):
    """
    Method loads the MURA data filtered by the specified body part two datasets split by class.
    Can be used to train CycleGANs.
    body_parts: List of body parts to work with. Check MURA documentation for available body_parts.
    tfds_path: Path to tensorflow datasets directory.
    batch_size: Batch size for the data loader.
    crop_size: Final image size that will be cropped to.
    load_size: The image will be loaded with this size.
    special_normalisation: Can be any normalisation from keras preprocessing (e.g. inception_preprocessing)
    transformed: If True, the transformed images will be loaded. If False, the original images will be loaded.
    """
    A_train, B_train, A_valid, B_valid, A_test, B_test = get_split_dataset_paths(body_parts, tfds_path, transformed)
    A_B_dataset, len_dataset_train = make_zip_dataset(A_train, B_train, batch_size, load_size,
                                                      crop_size, training=True, repeat=False,
                                                      special_normalisation=special_normalisation)

    A_B_dataset_valid, _ = make_zip_dataset(A_valid, B_valid, batch_size, load_size,
                                            crop_size, training=False, repeat=True,
                                            special_normalisation=special_normalisation)

    A_B_dataset_test, _ = make_zip_dataset(A_test, B_test, batch_size, load_size,
                                           crop_size, training=False, repeat=True,
                                           special_normalisation=special_normalisation)
    return A_B_dataset, A_B_dataset_valid, A_B_dataset_test, len_dataset_train


def get_mura_ds_by_body_part(body_parts, dataset_root, batch_size, crop_size, load_size, special_normalisation=None,
                             transformed=True):
    """
    Method loads the MURA data filtered by the specified body part in one dataset. Can be used to train classifiers.
    body_parts: List of body parts to work with. Check MURA documentation for available body_parts.
    dataset_root: Path to datasets directory where MURA dataset lies
    batch_size: Batch size for the data loader.
    crop_size: Final image size that will be cropped to.
    load_size: The image will be loaded with this size.
    special_normalisation: Can be any normalisation from keras preprocessing (e.g. inception_preprocessing)
    transformed: If True, the images will be loaded from the transformed dataset (MURA-v1.1-transformed)
    """
    A_train, B_train, A_valid, B_valid, A_test, B_test = get_split_dataset_paths(body_parts, dataset_root, transformed)
    A_B_dataset, len_dataset_train = make_concat_dataset(A_train, B_train, batch_size,
                                                         load_size,
                                                         crop_size, training=True, shuffle=True,
                                                         special_normalisation=special_normalisation)

    A_B_dataset_valid, _ = make_concat_dataset(A_valid, B_valid, batch_size,
                                               load_size,
                                               crop_size, training=False,
                                               special_normalisation=special_normalisation)

    A_B_dataset_test, _ = make_concat_dataset(A_test, B_test, batch_size, load_size,
                                              crop_size, training=False,
                                              special_normalisation=special_normalisation)
    return A_B_dataset, A_B_dataset_valid, A_B_dataset_test, len_dataset_train


def get_mura_test_ds_by_body_part_split_class(body_parts, tfds_path, batch_size, crop_size, load_size,
                                              special_normalisation=None):
    """
    Method loads the TEST MURA data filtered by the specified body part two datasets split by class.
    Can be used to test CycleGANs.
    body_parts: List of body parts to work with. Check MURA documentation for available body_parts.
    tfds_path: Path to tensorflow datasets directory.
    batch_size: Batch size for the data loader.
    crop_size: Final image size that will be cropped to.
    load_size: The image will be loaded with this size.
    special_normalisation: Can be any normalisation from keras preprocessing (e.g. inception_preprocessing)
    """
    A_train, B_train, A_valid, B_valid, A_test, B_test = get_split_dataset_paths(body_parts, tfds_path)

    # merge train and valid to not explude images
    A_train.extend(A_valid)
    B_train.extend(B_valid)

    A_dataset = make_dataset(A_train, batch_size, load_size, crop_size, training=False,
                             drop_remainder=True, shuffle=True, repeat=1, special_normalisation=special_normalisation)

    B_dataset = make_dataset(B_train, batch_size, load_size, crop_size, training=False,
                             drop_remainder=True, shuffle=True, repeat=1, special_normalisation=special_normalisation)

    A_dataset_test = make_dataset(A_test, batch_size, load_size, crop_size, training=False, drop_remainder=True,
                                  shuffle=False, repeat=1, special_normalisation=special_normalisation)
    B_dataset_test = make_dataset(B_test, batch_size, load_size, crop_size, training=False, drop_remainder=True,
                                  shuffle=False, repeat=1, special_normalisation=special_normalisation)

    return A_dataset, B_dataset, A_dataset_test, B_dataset_test
