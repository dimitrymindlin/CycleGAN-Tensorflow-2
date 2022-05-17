import numpy as np
import skimage.io as iio
import os

from matplotlib import pyplot as plt

from imlib import dtype
from imlib.transform import immerge


def generate_image(args, clf, A, B, A2B, B2A,
                   execution_id, ep, batch_count,
                   A_holder=None,
                   B_holder=None):
    if args.attention_type != "none":  # Save with attention
        if args.dataset == "mura":
            imgs = [A, A_holder.attention, A_holder.transformed_part, A2B,
                    B, B_holder.attention, B_holder.transformed_part, B2A]
            save_mura_images_with_attention(imgs, clf, args.dataset, execution_id, ep, batch_count)
        else:
            save_images_with_attention(A_holder, A2B, B_holder, B2A,
                                       clf, args.dataset, execution_id, ep, batch_count,
                                       args.attention_type)
    else:  # Save without attention
        if args.dataset == "mura":
            imgs = [A, A2B, B, B2A]
            save_mura_images(imgs, clf, args.dataset, execution_id, ep, batch_count)
        else:
            save_images(A, A2B, B, B2A, args.dataset, execution_id, ep, batch_count)


def save_mura_images(imgs, clf, dataset, execution_id, ep_cnt, batch_count):
    r, c = 2, 2
    titles = ['Original', 'Translated',
              'Original', 'Translated']
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
                    f'{titles[j]} (T: {correct_classification[cnt]} | P: {classification[cnt]})')
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
    titles = ['Original', 'Attention', 'Translated', 'Output']
    classification = [['Normal', 'Abnormal'][int(np.argmax(clf.predict(x)))] for x in imgs]
    gen_imgs = np.concatenate(imgs)
    gen_imgs = 0.5 * gen_imgs + 0.5
    correct_classification = ['Normal', 'Normal', 'Normal', 'Abnormal',
                              'Abnormal', 'Abnormal', 'Abnormal', 'Normal']
    fig, axs = plt.subplots(r, c, figsize=(30, 20))
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt][:, :, 0], cmap='gray')
            if j in [0, 3]:
                axs[i, j].set_title(
                    f'{titles[j]} (T: {correct_classification[cnt]} | P: {classification[cnt]})')
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


def save_images_with_attention(A_attention_image, A2B, B_attention_image, B2A, clf, dataset,
                               execution_id, ep_cnt, batch_count, attention_type):
    if attention_type == "spa-gan":
        img = immerge(
            np.concatenate([A_attention_image.img, A_attention_image.attention, A_attention_image.enhanced_img, A2B,
                            B_attention_image.img, B_attention_image.attention, B_attention_image.enhanced_img, B2A],
                           axis=0), n_rows=2)
    else:  # attention-gan
        img = immerge(
            np.concatenate([A_attention_image.img, A_attention_image.attention, A_attention_image.transformed_part, A2B,
                            B_attention_image.img, B_attention_image.attention, B_attention_image.transformed_part,
                            B2A],
                           axis=0), n_rows=2)

    img_folder = f'output_{dataset}/{execution_id}/images'
    if clf:
        classification = [['A', 'B'][int(np.argmax(clf.predict(x)))] for x in [A_attention_image.img, A2B,
                                                                               B_attention_image.img, B2A]]
        AB_correct, BA_correct = False, False
        if classification[0] == 'A' and classification[1] == "B":
            AB_correct = True
        if classification[2] == 'B' and classification[3] == "A":
            BA_correct = True
        try:
            imwrite(img,
                    f"{img_folder}/%d_%d_AB:{AB_correct}_BA:{BA_correct}.png" % (
                        ep_cnt, batch_count))
        except (AssertionError, AttributeError, OSError) as e:
            print(f"Wasn't able to print image {ep_cnt}_{batch_count}")
            print(e)
    else:
        try:
            imwrite(img,
                    f"{img_folder}/%d_%d.png" % (
                        ep_cnt, batch_count))
        except (AssertionError, AttributeError, OSError) as e:
            print(f"Wasn't able to print image {ep_cnt}_{batch_count}")
            print(e)


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


def plot_any_img(img):
    plt.imshow(np.squeeze(img), vmin=np.min(img), vmax=np.max(img))
    plt.show()


show = iio.show