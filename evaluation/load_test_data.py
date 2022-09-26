import tensorflow as tf
import tensorflow_datasets as tfds


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
