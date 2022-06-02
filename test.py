import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tf2lib as tl
import tensorflow_datasets as tfds
import data
import module

# ==============================================================================
# =                                   param                                    =
# ==============================================================================
from evaluation.kid import KID

gan_model_ts = "2022-05-23--18.32"
py.arg('--experiment_dir', default=f"checkpoints/gans/horse2zebra/{gan_model_ts}")
py.arg('--batch_size', type=int, default=32)
py.arg('--print_images', type=bool, default=False)
py.arg('--crop_size', type=int, default=256)
args = py.args()
# args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
# args.__dict__.update(test_args.__dict__)

# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# data
AUTOTUNE = tf.data.AUTOTUNE
dataset, metadata = tfds.load('cycle_gan/horse2zebra',
                              with_info=True, as_supervised=True)

train_horses, train_zebras = dataset['trainA'], dataset['trainB']
test_horses, test_zebras = dataset['testA'], dataset['testB']

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


# normalizing the images to [-1, 1]
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
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_zebras = test_zebras.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

OUTPUT_CHANNELS = 3
# model
G_A2B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))
G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))

# resotre
# tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(args.experiment_dir, 'checkpoints')).restore()
tl.Checkpoint(dict(generator_g=G_A2B, generator_f=G_B2A), py.join(args.experiment_dir, 'checkpoints')).restore()

print("Loaded Model :)")
print()


@tf.function
def sample_A2B(A):
    A2B = G_A2B(A, training=False)
    A2B2A = G_B2A(A2B, training=False)
    return A2B, A2B2A


@tf.function
def sample_B2A(B):
    B2A = G_B2A(B, training=False)
    B2A2B = G_A2B(B2A, training=False)
    return B2A, B2A2B


# run
save_dir = py.join(args.experiment_dir, 'samples_testing')
py.mkdir(save_dir)
clf = tf.keras.models.load_model(f"checkpoints/inception_horse2zebra_512/model", compile=False)
oracle = tf.keras.models.load_model(f"checkpoints/resnet50_horse2zebra_256/model", compile=False)


def testing_loop(dataset_A, dataset_B, translation_name, only_target_domain=False):
    print(f"Starting testing {translation_name}")
    len_dataset = 0
    kid = KID(image_size=args.crop_size)
    #ssim_count = 0
    #psnr_count = 0
    tcv = 0
    real_images = []
    translated_images = []
    y_pred_translated = []
    y_pred_oracle = []
    for img_batch_A, img_batch_B in tqdm.tqdm(tf.data.Dataset.zip((dataset_A, dataset_B))):
        if translation_name == "A2B":
            translated_img_batch, cycled_img_batch = sample_A2B(img_batch_A)
            for img_i, translated_i, cycled_i in zip(img_batch_A, translated_img_batch, cycled_img_batch):
                real_images.append(tf.squeeze(img_i))
                translated_images.append(tf.squeeze(translated_i))
                y_pred_translated.append(
                    int(np.argmax(clf(tf.expand_dims(tf.image.resize(translated_i, [512, 512]), axis=0)))))
                y_pred_oracle.append(
                    int(np.argmax(oracle(tf.expand_dims(translated_i, axis=0)))))
                len_dataset += 1
        else:
            translated_img_batch, cycled_img_batch = sample_B2A(img_batch_B)
        for img_i, translated_i, cycled_i in zip(img_batch, translated_img_batch, cycled_img_batch):
            real_images.append(tf.squeeze(img_i))
            translated_images.append(tf.squeeze(translated_i))
            #img_i = img_i.numpy()
            #translated_i = translated_i.numpy()
            y_pred_translated.append(
                int(np.argmax(clf(tf.expand_dims(tf.image.resize(translated_i, [512, 512]), axis=0)))))
            y_pred_oracle.append(
                int(np.argmax(oracle(tf.expand_dims(translated_i, axis=0)))))
            """if args.print_images:
                img = np.concatenate([A_i, A2B_i, A2B2A_i], axis=1)
                im.imwrite(img, py.join(save_dir, py.name_ext(A_img_paths_test[i])))"""
            #ssim_count += structural_similarity(img_i, translated_i, channel_axis=2, data_range=2)
            #psnr_count += peak_signal_noise_ratio(img_i, translated_i, data_range=2)
            len_dataset += 1
    if translation_name == "A2B":
        tcv = sum(y_pred_translated) / len_dataset
        similar_predictions_count = sum(x == y == 1 for x, y in zip(y_pred_translated, y_pred_oracle))
        os = (1 / len_dataset) * similar_predictions_count
    else:
        tcv = (len_dataset - sum(y_pred_translated)) / len_dataset
        similar_predictions_count = sum(x == y == 0 for x, y in zip(y_pred_translated, y_pred_oracle))
        os = (1 / len(y_pred_translated)) * similar_predictions_count
    kid.update_state(tf.convert_to_tensor(real_images), tf.convert_to_tensor(translated_images))
    print(f"Results for {translation_name}")
    #print(f"SSIM: ", ssim_count / len_dataset)
    #print(f"PSNR: ", psnr_count / len_dataset)
    print(f"KID: ", kid.result())
    print(f"TCV:", tcv)
    print(f"OS :", os)
    return kid.result(), tcv, os


kid, tcv, os = testing_loop(test_horses, test_horses, "A2B")
kid, tcv, os = testing_loop(test_zebras, test_horses, "B2A")
