import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tf2lib as tl
import tensorflow_addons as tfa
import pylib as py
import tensorflow_datasets as tfds


def make_dataset(img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=True, repeat=1,
                 labels=None, special_normalisation=None):
    if training:
        @tf.function
        def _map_fn(img, label=None):  # preprocessing
            img = tf.cast(img, tf.float32)
            img = tfa.image.equalize(img)
            img = tf.image.random_flip_left_right(img)
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
            #img = tfa.image.equalize(img)
            img = tf.image.resize_with_pad(img, crop_size, crop_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            if not special_normalisation:
                img = img / 255.0 * 2 - 1
            else:
                img = special_normalisation(img)
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


def make_zip_dataset(A_img_paths, B_img_paths, batch_size, load_size, crop_size, training, shuffle=False, repeat=False):
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

    A_dataset = make_dataset(A_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True,
                             shuffle=shuffle, repeat=1)
    B_dataset = make_dataset(B_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True,
                             shuffle=shuffle, repeat=1)

    A_B_dataset = tf.data.Dataset.zip((A_dataset, B_dataset))
    len_dataset = max(len(A_img_paths), len(B_img_paths)) // batch_size
    return A_dataset, B_dataset
    return A_B_dataset, len_dataset


def make_concat_dataset(A_img_paths, B_img_paths, batch_size, load_size, crop_size, training, shuffle=True,
                        repeat=False, special_normalisation=None):
    dataset_length = len(A_img_paths) + len(B_img_paths)
    class_labels = [(1, 0) for _ in range(len(A_img_paths))]
    class_labels.extend([(0, 1) for _ in range(len(B_img_paths))])
    A_img_paths.extend(B_img_paths)  # becoming all_image_paths
    all_image_paths = A_img_paths
    return make_dataset(all_image_paths, batch_size, load_size, crop_size, training, drop_remainder=True,
                        shuffle=shuffle, repeat=repeat, labels=class_labels,
                        special_normalisation=special_normalisation), dataset_length


class ItemPool:

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.items = []

    def __call__(self, in_items):
        # `in_items` should be a batch tensor

        if self.pool_size == 0:
            return in_items

        out_items = []
        for in_item in in_items:
            if len(self.items) < self.pool_size:
                self.items.append(in_item)
                out_items.append(in_item)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, len(self.items))
                    out_item, self.items[idx] = self.items[idx], in_item
                    out_items.append(out_item)
                else:
                    out_items.append(in_item)
        return tf.stack(out_items, axis=0)


def get_mura_data_paths():
    # To get the filenames for a task
    def to_categorical(x, y):
        y = [0 if x == 'negative' else 1 for x in y]
        y = tf.keras.utils.to_categorical(y)
        x, y = shuffle(x, y)
        return x, y

    def filenames(part, train=True):
        root = '../tensorflow_datasets/downloads/cjinny_mura-v11/'
        #root = '/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/'
        if train:
            csv_path = "../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/train_image_paths.csv"
            #csv_path = "/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/train_image_paths.csv"
        else:
            csv_path = "../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/valid_image_paths.csv"
            #csv_path = "/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/valid_image_paths.csv"

        with open(csv_path, 'rb') as F:
            d = F.readlines()
            if part == 'all':
                imgs = [root + str(x, encoding='utf-8').strip() for x in d]
            else:
                imgs = [root + str(x, encoding='utf-8').strip().replace("MURA-v1.1", "MURA-v1.1") for x in d
                        if
                        str(x, encoding='utf-8').strip().split('/')[2] in part]

        # imgs= [x.replace("/", "\\") for x in imgs]
        labels = [x.split('_')[-1].split('/')[0] for x in imgs]
        return imgs, labels

    part = 'XR_WRIST'  # part to work with
    train_x, train_y = filenames(part=part)  # train data
    test_x, test_y = filenames(part=part, train=False)  # test data
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2,
                                                          random_state=42)  # split train and valid data

    train_x, train_y = to_categorical(train_x, train_y)
    valid_x, valid_y = to_categorical(valid_x, valid_y)
    test_x, test_y = to_categorical(test_x, test_y)

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def get_dataset_paths(args):
    if args.dataset == "mura":
        # A = 0 = negative, B = 1 = positive
        train_x, train_y, valid_x, valid_y, test_x, test_y = get_mura_data_paths()
        A_img_paths = [filename for filename in train_x if "negative" in filename]
        B_img_paths = [filename for filename in train_x if "positive" in filename]
        A_img_paths_test = [filename for filename in test_x if "negative" in filename]
        B_img_paths_test = [filename for filename in test_x if "positive" in filename]
    else:
        A_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, 'trainA'), '*.jpg')
        B_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, 'trainB'), '*.jpg')
        A_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testA'), '*.jpg')
        B_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testB'), '*.jpg')
    return A_img_paths, B_img_paths, A_img_paths_test, B_img_paths_test


def load_tfds_dataset(dataset_name, img_size):
    AUTOTUNE = tf.data.AUTOTUNE
    dataset, metadata = tfds.load(f'cycle_gan/{dataset_name}',
                                  with_info=True, as_supervised=True)

    train_horses, train_zebras = dataset['trainA'], dataset['trainB']
    test_horses, test_zebras = dataset['testA'], dataset['testB']
    BUFFER_SIZE = 1000
    len_dataset = len(train_zebras)
    BATCH_SIZE = 1
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

    def preprocess_image_train(image, label):
        image = random_jitter(image)
        image = normalize(image)
        return image

    def preprocess_image_test(image, label):
        image = normalize(image)
        return image

    train_horses = train_horses.cache().map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    train_zebras = train_zebras.cache().map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    test_horses = test_horses.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    test_zebras = test_zebras.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    return train_horses, train_zebras, test_horses, test_zebras, len_dataset