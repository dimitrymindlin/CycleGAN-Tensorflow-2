import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

import pylib as py
from attention_maps import add_attention_maps
from tf2lib_local.data import disk_image_batch_dataset


def make_dataset(img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=True, repeat=1,
                 labels=None, special_normalisation=None):
    if training:
        @tf.function
        def _map_fn(img, label=None):  # preprocessing
            img = tf.cast(img, tf.float32)
            img = tf.image.random_flip_left_right(img)
            if np.shape(load_size)[0] > 1:
                img = tf.image.resize_with_pad(img, load_size[0], load_size[1],
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                img = tf.image.random_crop(img, [crop_size[0], crop_size[1], tf.shape(img)[-1]])
            else:
                img = tf.image.resize_with_pad(img, load_size, load_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])
            if not special_normalisation:
                img = img / 255.0 * 2 - 1
            else:
                img = special_normalisation(img)
            if label is not None:
                return img, label
            return img
    else:
        @tf.function
        def _map_fn(img, label=None):  # preprocessing
            img = tf.cast(img, tf.float32)
            # img = tfa.image.equalize(img)
            if np.shape(crop_size)[0] > 1:
                img = tf.image.resize_with_pad(img, crop_size[0], crop_size[1],
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            else:
                img = tf.image.resize_with_pad(img, crop_size, crop_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            if not special_normalisation:
                img = img / 255.0 * 2 - 1
            else:
                img = special_normalisation(img)
            if label is not None:
                return img, label
            return img

    return disk_image_batch_dataset(img_paths,
                                    batch_size,
                                    drop_remainder=drop_remainder,
                                    map_fn=_map_fn,
                                    shuffle=shuffle,
                                    repeat=repeat,
                                    labels=labels)


def make_zip_dataset(A_img_paths, B_img_paths, batch_size, load_size, crop_size, training, shuffle=False, repeat=False):
    # zip two datasets aligned by the longer one (for GAN training)
    if repeat:
        A_repeat = B_repeat = None  # cycle both
    else:
        if len(A_img_paths) >= len(B_img_paths):
            A_repeat = 1
            B_repeat = None  # cycle the shorter one
        else:
            A_repeat = None  # cycle the shorter one
            B_repeat = 1

    A_dataset = make_dataset(A_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True,
                             shuffle=shuffle, repeat=A_repeat)
    B_dataset = make_dataset(B_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True,
                             shuffle=shuffle, repeat=B_repeat)

    A_B_dataset = tf.data.Dataset.zip((A_dataset, B_dataset))
    len_dataset = max(len(A_img_paths), len(B_img_paths)) // batch_size
    return A_B_dataset, len_dataset


def make_concat_dataset(A_img_paths, B_img_paths, batch_size, load_size, crop_size, training, shuffle=True,
                        repeat=1, special_normalisation=None):
    # concat two datasets (for clf training)
    dataset_length = len(A_img_paths) + len(B_img_paths)
    class_labels = [(1, 0) for _ in range(len(A_img_paths))]
    class_labels.extend([(0, 1) for _ in range(len(B_img_paths))])
    A_img_paths.extend(B_img_paths)  # becoming all_image_paths
    all_image_paths = A_img_paths
    return make_dataset(all_image_paths, batch_size, load_size, crop_size, training, drop_remainder=True,
                        shuffle=shuffle, repeat=1, labels=class_labels,
                        special_normalisation=special_normalisation), dataset_length


def get_dataset_paths(args):
    A_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, 'trainA'), '*.jpg')
    B_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, 'trainB'), '*.jpg')
    A_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testA'), '*.jpg')
    B_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testB'), '*.jpg')
    return A_img_paths, B_img_paths, A_img_paths_test, B_img_paths_test


def load_tfds_dataset(dataset_name, img_size, clf=None, gradcam=None):
    AUTOTUNE = tf.data.AUTOTUNE
    dataset, metadata = tfds.load(f'cycle_gan/{dataset_name}',
                                  with_info=True, as_supervised=True)

    A_train, B_train = dataset['trainA'], dataset['trainB']  # A=horses, B=zebras
    A_test, B_test = dataset['testA'], dataset['testB']

    BUFFER_SIZE = 1000
    len_dataset_train = max(len(B_train), len(A_train))
    BATCH_SIZE = 1
    if np.shape(img_size)[0] > 1:
        IMG_HEIGHT = img_size[0]
        IMG_WIDTH = img_size[1]
    else:
        IMG_WIDTH = img_size
        IMG_HEIGHT = img_size

    def random_crop(image):
        cropped_image = tf.image.random_crop(
            image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

        return cropped_image

    # normalizing the images to [-1, 1]
    def normalize(image):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = (image / 127.5) - 1
        return image

    def random_jitter(image):
        # resizing to 286 x 286 x 3
        image = tf.image.resize(image, [IMG_HEIGHT + 30, IMG_WIDTH + 30],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # randomly cropping to 256 x 256 x 3
        image = random_crop(image)

        # random mirroring
        image = tf.image.random_flip_left_right(image)

        return image

    def preprocess_image_train(ds_tuple, mask):
        # image = random_jitter(image)
        # normalize img and return img + mask (drop label)
        img = ds_tuple[0]
        img = normalize(img)
        final_tuple = (img, mask)
        return final_tuple

    def preprocess_image_test(image, label):
        image = normalize(image)
        return image

    A_train, B_train = add_attention_maps(A_train, B_train, gradcam, IMG_HEIGHT, IMG_WIDTH)

    A_train = A_train.cache().map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    B_train = B_train.cache().map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    A_test = A_test.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    B_test = B_test.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    A_B_dataset = tf.data.Dataset.zip(((A_train, B_train)))
    A_B_dataset_test = tf.data.Dataset.zip(((A_test, B_test)))

    return A_B_dataset, A_B_dataset_test, len_dataset_train


def get_celeba_smiling_non_smiling_paths(TFDS_PATH):
    # Here we import dataset to python and assign it to data variable
    data = pd.read_csv(f"{TFDS_PATH}/celeba/list_attr_celeba.csv")

    # We make list of smiling and non-smiling celebrities.
    smiling = []
    non_smiling = []
    for value, img_id in zip(data.Smiling.values, data.image_id.values):
        if value == 1:
            smiling.append(img_id)
        else:
            non_smiling.append(img_id)

    # extend both lists to have full img path
    smiling = [f"{TFDS_PATH}/celeba/img_align_celeba/img_align_celeba/{img_id}" for img_id in smiling]
    non_smiling = [f"{TFDS_PATH}/celeba/img_align_celeba/img_align_celeba/{img_id}" for img_id in non_smiling]
    return smiling, non_smiling


def get_calaba_zip_dataset(TFDS_PATH, crop_size):
    # A is smiling, B is non smiling
    smiling_paths, non_smiling_paths = get_celeba_smiling_non_smiling_paths(TFDS_PATH)
    A_img_paths, A_img_paths_test = train_test_split(smiling_paths, test_size=0.2, random_state=42)
    B_img_paths, B_img_paths_test = train_test_split(non_smiling_paths, test_size=0.2, random_state=42)
    A_B_datset_train, len_dataset_train = make_zip_dataset(A_img_paths, B_img_paths, 1, crop_size, crop_size, True,
                                                           shuffle=False, repeat=False)
    A_B_datset_test, _ = make_zip_dataset(A_img_paths_test, B_img_paths_test, 1, crop_size, crop_size, True,
                                          shuffle=False, repeat=False)

    return A_B_datset_train, A_B_datset_test, len_dataset_train
