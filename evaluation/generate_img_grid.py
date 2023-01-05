import tqdm
import tensorflow as tf
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from attention_strategies.attention_gan import attention_gan_single
from attention_strategies.no_attention import no_attention_single
from evaluation.tcv_os import get_predicted_class_label
from imlib.image_holder import ImageHolder
from evaluation.compute_img_difference import get_difference_img

titles_list = ["Original", "ABC-GAN", "Diff", "Attention", "GTF", "Diff", "CycleGAN", "Diff"]


def get_empty_x_coordinate(axs, fig):
    # Get a list of the images in the figure
    images = [image for ax in axs.flat for image in ax.get_images()]

    # Get the indices of the subplots that do not have an image
    empty_subplot_indices = [ax.index for ax in axs.flat if ax not in images]

    # Get the 2D coordinates of the subplots that do not have an image
    empty_subplot_coords = [fig.subplot_index(index) for index in empty_subplot_indices]
    return empty_subplot_coords[0][0]


def plot_img_grid(original_imgs_list, translated_imgs_list, attention_maps_list, is_counterfactual_list,
                  last_model=False, axs=None):
    if axs is None:
        fig, axs = plt.subplots(len(original_imgs_list), 4 + 4)
    for y_counter, (original, translated, attention_map, is_counterfactual) in enumerate(
            zip(original_imgs_list, translated_imgs_list,
                attention_maps_list, is_counterfactual_list)):

        # Compute SSIM between the two images
        difference = get_difference_img(original, translated)
        grid_row = [original, translated, difference, attention_map]

        for idx, img in enumerate(grid_row):
            ### Tweaking x_position of images to add images to the axs object.
            x_counter = idx
            if last_model:
                if idx == 0:
                    continue
                x_counter += 5  # last model
                if x_counter == 8:
                    break

            elif attention_map is None:
                if idx == 0:
                    continue
                x_counter += 3  # middle model (ganterfactual has no attention)

            try:
                img = tf.squeeze(img)
            except ValueError:
                continue  # when img is none (e.g. no attention img)
            # Convert [-1,1] array to [0,255] int array.
            img = img.numpy()
            img = ((img * 0.5 + 0.5) * 255).astype("uint8")
            axs[y_counter][x_counter].imshow(img, cmap='gray', vmin=np.min(img), vmax=np.max(img))
            if idx == 1:
                color = "green" if is_counterfactual else "red"
                rect = Rectangle((0, 0), *img.shape[:2], fill=False, edgecolor=color, linewidth=4)
                axs[y_counter][x_counter].add_patch(rect)
            axs[y_counter][x_counter].axis("off")
            if y_counter == 0:
                axs[y_counter][x_counter].set_title(titles_list[x_counter])
    if last_model:
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0.1)
        plt.show()
        print("Hey")
    else:
        return axs


def translate_and_predict(img_holder, generator, args, clf, training, class_label):
    # Translate image using current generator
    if args.attention_type == "attention-gan-original":
        translated_img, _ = attention_gan_single(img_holder.img, generator, None, img_holder.attention,
                                                 img_holder.background, training)
    else:
        translated_img = no_attention_single(img_holder.img, generator, None, training)

    # Predict images with CLF
    # To predict, scale to 512,512 and batch
    translated_i_batched = tf.expand_dims(tf.image.resize(translated_img[0], [512, 512]), axis=0)
    clf_prediction = get_predicted_class_label(args, translated_i_batched, clf)
    if_counterfactual = True if class_label != clf_prediction else False
    return translated_img, if_counterfactual


def generate_images_for_grid(args, dataset, clf, generator, gradcam, class_label,
                             training=False, num_images=6, last_model=False, axs=None):
    """
    Method uses generators to translate images to new domain and uses the classifier to predict the label of the
    translated image. Then saves the translated images along with the attention map and the notion whether it is a
    counterfactual or not.
    """
    if args.attention_type != "none":
        use_attention = True
    else:
        use_attention = False

    original_images = []
    translated_images = []
    attention_maps = []
    is_counterfactual_list = []

    for batch_i, img_batch in enumerate(tqdm.tqdm(dataset, desc='Translating images')):
        # Save img when enough images are collected
        if len(translated_images) == num_images:
            axs = plot_img_grid(original_images, translated_images, attention_maps, is_counterfactual_list, last_model,
                                axs=axs)
            original_images = []
            translated_images = []
            attention_maps = []
            is_counterfactual_list = []
            if batch_i > 6:  # Only make 2 plots...
                return axs
        # Only use samples where the clf prediction is correct for the input img.
        original_img_batched = tf.expand_dims(tf.image.resize(img_batch[0], [512, 512]), axis=0)
        original_prediction = get_predicted_class_label(args, original_img_batched, clf)
        if original_prediction != class_label:
            continue

        img_holder = ImageHolder(img_batch, args, class_label, gradcam=gradcam, use_attention=use_attention)
        attention_maps.append(img_holder.attention)
        original_images.append(img_holder.img)
        translated_img, if_counterfactual = translate_and_predict(img_holder, generator, args, clf, training,
                                                                  class_label)
        translated_images.append(translated_img)
        is_counterfactual_list.append(if_counterfactual)
