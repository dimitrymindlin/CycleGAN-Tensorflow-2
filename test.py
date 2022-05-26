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
#args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
#args.__dict__.update(test_args.__dict__)

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
#tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(args.experiment_dir, 'checkpoints')).restore()
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
i = 0
kid = KID(image_size=args.crop_size)
ssim_count = 0
psnr_count = 0
real_images = []
generated_images = []
for batch_count, (A, B) in enumerate(tqdm.tqdm(tf.data.Dataset.zip((test_horses, test_zebras)))):
    A2B, A2B2A = sample_A2B(A)
    B2A, _ = sample_B2A(B)
    for A_i, A2B_i, A2B2A_i, B2A_i in zip(A, A2B, A2B2A, B2A):
        real_images.append(tf.squeeze(A_i))
        generated_images.append(tf.squeeze(B2A_i))
        A_i = A_i.numpy()
        A2B_i = A2B_i.numpy()
        A2B2A_i = A2B2A_i.numpy()
        """if args.print_images:
            img = np.concatenate([A_i, A2B_i, A2B2A_i], axis=1)
            im.imwrite(img, py.join(save_dir, py.name_ext(A_img_paths_test[i])))"""
        ssim_count += structural_similarity(A_i, A2B_i, channel_axis=2)
        psnr_count += peak_signal_noise_ratio(A_i, A2B_i)
        i += 1
kid.update_state(tf.convert_to_tensor(real_images), tf.convert_to_tensor(generated_images))
print("A2B SSIM: ", ssim_count / i)
print("A2B PSNR: ", psnr_count / i)
print("KID for A: ", kid.result())


"""i = 0
kid.reset_state()
ssim_count = 0
psnr_count = 0
real_images = []
generated_images = []
for B in tqdm.tqdm(B_dataset_test, desc='Test Loop', total=int(len_B_test / args.batch_size)):
    B2A, B2A2B = sample_B2A(B)
    real_images.append(B)
    generated_images.append(B2A)
    for B_i, B2A_i, B2A2B_i in tqdm.tqdm(zip(B, B2A, B2A2B), total=args.batch_size):
        if args.print_images:
            img = np.concatenate([B_i.numpy(), B2A_i.numpy(), B2A2B_i.numpy()], axis=1)
            im.imwrite(img, py.join(save_dir, py.name_ext(B_img_paths_test[j])))
        ssim_count += structural_similarity(B_i, B2A_i, channel_axis=2)
        psnr_count += peak_signal_noise_ratio(B_i, B2A_i)
        i += 1
print("A2B SSIM: ", ssim_count / len_B_test)
print("A2B PSNR: ", psnr_count / len_B_test)
print("KID for A: ", kid.result())"""