import tensorflow as tf
import tensorflow_datasets as tfds
from mura.tfds_from_disc import get_mura_test_ds_by_body_part_split_class
from rsna import get_rsna_TEST_ds_split_class

from config import ROOT_DIR

TFDS_PATH = f"{ROOT_DIR}/../tensorflow_datasets"  # Path to datasets


def load_tfds_test_data(dataset_name):
    AUTOTUNE = tf.data.AUTOTUNE
    dataset, metadata = tfds.load(f'cycle_gan/{dataset_name}',
                                  with_info=True, as_supervised=True)

    train_horses, train_zebras = dataset['trainA'], dataset['trainB']
    test_horses, test_zebras = dataset['testA'], dataset['testB']

    BUFFER_SIZE = 100
    BATCH_SIZE = 1

    def normalize(image):
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1
        return image

    def preprocess_image_test(image, label):
        image = normalize(image)
        return image

    train_horses = train_horses.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    train_zebras = train_zebras.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    test_horses = test_horses.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().batch(BATCH_SIZE)

    test_zebras = test_zebras.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().batch(BATCH_SIZE)
    return train_horses, test_horses, train_zebras, test_zebras


def load_test_data_from_args(args):
    if args.dataset == "mura":
        A_dataset, B_dataset, A_dataset_test, B_dataset_test = get_mura_test_ds_by_body_part_split_class(
            args.body_parts,
            TFDS_PATH,
            args.batch_size,
            args.crop_size,
            args.crop_size,
            special_normalisation=None)
    elif args.dataset == "rsna":
        A_dataset, B_dataset, A_dataset_test, B_dataset_test = get_rsna_TEST_ds_split_class(TFDS_PATH,
                                                                                            args.batch_size,
                                                                                            args.crop_size,
                                                                                            args.crop_size,
                                                                                            special_normalisation=None,
                                                                                            channels=args.img_channels,
                                                                                            training=False)


    else:  # Horse2Zebra / Apple2Orange
        A_dataset, A_dataset_test, B_dataset, B_dataset_test = load_tfds_test_data(args.dataset)
    print("A_dataset", len(A_dataset))
    print("A_dataset_test", len(A_dataset_test))
    print("B_dataset", len(B_dataset))
    print("B_dataset_test", len(B_dataset_test))
    return A_dataset, A_dataset_test, B_dataset, B_dataset_test
