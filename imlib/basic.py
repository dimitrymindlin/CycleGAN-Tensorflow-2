import numpy as np
import skimage.io as iio
import os

from matplotlib import pyplot as plt

from imlib import dtype
from imlib.transform import immerge


def save_mura_images(imgs, clf, dataset, execution_id, ep_cnt, batch_count):
    r, c = 2, 2
    titles = ['Original', 'Translated', 'Original', 'Translated']
    classification = [['Normal', 'Abnormal'][int(np.argmax(clf.predict(x)))] for x in imgs]
    gen_imgs = np.concatenate(imgs)
    gen_imgs = 0.5 * gen_imgs + 0.5
    if dataset == "mura":
        correct_classification = ['Normal', 'Abnormal',
                                  'Abnormal', 'Normal']
    else:
        correct_classification = ['A', 'B',
                                  'B', 'A']
    fig, axs = plt.subplots(r, c, figsize=(30, 20))
    cnt = 0
    for i in range(r):
        for j in range(c):
            if dataset == "mura":
                axs[i, j].imshow(gen_imgs[cnt][:, :, 0], cmap='gray')
            else:
                axs[i, j].imshow(gen_imgs[cnt][:, :, 0])
            if j in [0, 3]:
                axs[i, j].set_title(
                    f'{titles[j]} T: ({correct_classification[cnt]} | P: {classification[cnt]})')
            else:
                axs[i, j].set_title(f'{titles[j]}')
            axs[i, j].axis('off')
            cnt += 1
    img_folder = f'output_{dataset}/{execution_id}/images'
    os.makedirs(img_folder, exist_ok=True)
    fig.savefig(f"{img_folder}/%d_%d.png" % (ep_cnt, batch_count))
    plt.close()


def save_mura_images_with_attention(imgs, clf, dataset, execution_id, ep_cnt, batch_count):
    r, c = 2, 4
    titles = ['Original', 'Attention', 'Input Image', 'Translated']
    classification = [['Normal', 'Abnormal'][int(np.argmax(clf.predict(x)))] for x in imgs]
    gen_imgs = np.concatenate(imgs)
    gen_imgs = 0.5 * gen_imgs + 0.5
    if dataset == "mura":
        correct_classification = ['Normal', 'Normal', 'Normal', 'Abnormal',
                                  'Abnormal', 'Abnormal', 'Abnormal', 'Normal']
    else:
        correct_classification = ['A', 'A', 'A', 'B',
                                  'B', 'A', 'A', 'A']
    fig, axs = plt.subplots(r, c, figsize=(30, 20))
    cnt = 0
    for i in range(r):
        for j in range(c):
            if dataset == "mura":
                axs[i, j].imshow(gen_imgs[cnt][:, :, 0], cmap='gray')
            else:
                axs[i, j].imshow(gen_imgs[cnt][:, :, 0])
            if j in [0, 3]:
                axs[i, j].set_title(
                    f'{titles[j]} T: ({correct_classification[cnt]} | P: {classification[cnt]})')
            else:
                axs[i, j].set_title(f'{titles[j]}')
            axs[i, j].axis('off')
            cnt += 1
    img_folder = f'output_{dataset}/{execution_id}/images'
    os.makedirs(img_folder, exist_ok=True)
    fig.savefig(f"{img_folder}/%d_%d.png" % (ep_cnt, batch_count))
    plt.close()


def save_images(A, A2B, B, B2A, dataset, execution_id, ep_cnt, batch_count):
    img = immerge(np.concatenate([A, A2B, B, B2A], axis=0), n_rows=2)
    img_folder = f'output_{dataset}/{execution_id}/images'
    imwrite(img, f"{img_folder}/%d_%d.png" % (ep_cnt, batch_count))


def save_images_with_attention(A, A_heatmap, A_attention, A2B, B, B_heatmap, B_attention, B2A, clf, dataset,
                               execution_id, ep_cnt, batch_count):
    img = immerge(
        np.concatenate([A, A_heatmap, A_attention, A2B, B, B_heatmap, B_attention, B2A], axis=0),
        n_rows=2)
    classification = [['A', 'B'][int(np.argmax(clf.predict(x)))] for x in [A, A2B, B, B2A]]
    AB_correct, BA_correct = False, False
    if classification[0] == 'A' and classification[1] == "B":
        AB_correct = True
    if classification[2] == 'B' and classification[3] == "A":
        BA_correct = True
    img_folder = f'output_{dataset}/{execution_id}/images'
    try:
        imwrite(img,
                f"{img_folder}/%d_%d_AB:{AB_correct}_BA:{BA_correct}.png" % (
                    ep_cnt, batch_count))
    except (AssertionError, AttributeError, OSError):
        print(f"Wasn't able to print image {ep_cnt}_{batch_count}")


def imread(path, as_gray=False, **kwargs):
    """Return a float64 image in [-1.0, 1.0]."""
    image = iio.imread(path, as_gray, **kwargs)
    if image.dtype == np.uint8:
        image = image / 127.5 - 1
    elif image.dtype == np.uint16:
        image = image / 32767.5 - 1
    elif image.dtype in [np.float32, np.float64]:
        image = image * 2 - 1.0
    else:
        raise Exception("Inavailable image dtype: %s!" % image.dtype)
    return image


def imwrite(image, path, quality=95, **plugin_args):
    """Save a [-1.0, 1.0] image."""
    iio.imsave(path, dtype.im2uint(image), **plugin_args)


def imshow(image):
    """Show a [-1.0, 1.0] image."""
    iio.imshow(dtype.im2uint(image))


show = iio.show
