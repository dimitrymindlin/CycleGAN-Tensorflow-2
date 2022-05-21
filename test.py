import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tf2lib as tl

import data
import module

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

gan_model_ts = "2022-05-16--20.46"
py.arg('--experiment_dir', default=f"checkpoints/gans/horse2zebra/{gan_model_ts}")
py.arg('--batch_size', type=int, default=32)
py.arg('--print_images', type=bool, default=False)
test_args = py.args()
args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)

# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# data
A_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testA'), '*.jpg')
len_A_test = len(A_img_paths_test)
B_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testB'), '*.jpg')
len_B_test = len(B_img_paths_test)
A_dataset_test = data.make_dataset(A_img_paths_test, args.batch_size, args.load_size, args.crop_size,
                                   training=False, drop_remainder=False, shuffle=False, repeat=1)
B_dataset_test = data.make_dataset(B_img_paths_test, args.batch_size, args.load_size, args.crop_size,
                                   training=False, drop_remainder=False, shuffle=False, repeat=1)

# model
G_A2B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))
G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))

# resotre
tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(args.experiment_dir, 'checkpoints')).restore()
print("Loaded Model :)")

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
ssim_count = 0
psnr_count = 0
for A in tqdm.tqdm(A_dataset_test, desc='Outer Loop', total=int(len_A_test / args.batch_size)):
    A2B, A2B2A = sample_A2B(A)
    for A_i, A2B_i, A2B2A_i in tqdm.tqdm(zip(A, A2B, A2B2A), total=args.batch_size):
        A_i = A_i.numpy()
        A2B_i = A2B_i.numpy()
        A2B2A_i = A2B2A_i.numpy()
        if args.print_images:
            img = np.concatenate([A_i, A2B_i, A2B2A_i], axis=1)
            im.imwrite(img, py.join(save_dir, py.name_ext(A_img_paths_test[i])))
        ssim_count += structural_similarity(A_i, A2B_i, channel_axis=2)
        psnr_count += peak_signal_noise_ratio(A_i, A2B_i)
        i += 1
print("A2B SSIM: ", ssim_count / len_A_test)
print("A2B PSNR: ", psnr_count / len_A_test)
print()

i = 0
ssim_count = 0
psnr_count = 0
for B in tqdm.tqdm(B_dataset_test, desc='Test Loop', total=int(len_B_test / args.batch_size)):
    B2A, B2A2B = sample_B2A(B)
    for B_i, B2A_i, B2A2B_i in tqdm.tqdm(zip(B, B2A, B2A2B), total=args.batch_size):
        if args.print_images:
            img = np.concatenate([B_i.numpy(), B2A_i.numpy(), B2A2B_i.numpy()], axis=1)
            im.imwrite(img, py.join(save_dir, py.name_ext(B_img_paths_test[j])))
        ssim_count += structural_similarity(B_i, B2A_i, channel_axis=2)
        psnr_count += peak_signal_noise_ratio(B_i, B2A_i)
        i += 1
print("A2B SSIM: ", ssim_count / len_B_test)
print("A2B PSNR: ", psnr_count / len_B_test)