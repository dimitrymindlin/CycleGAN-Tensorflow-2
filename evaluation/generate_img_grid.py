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


def plot_img_grid(original_imgs_list, translated_imgs_list, attention_maps_list, is_counterfactual_list):
    fig, axs = plt.subplots(len(original_imgs_list), 4)
    for y_counter, (original, translated, attention_map, is_counterfactual) in enumerate(
            zip(original_imgs_list, translated_imgs_list,
                attention_maps_list, is_counterfactual_list)):

        difference = get_difference_img(original, translated)
        grid_row = [original, translated, difference, attention_map]
        # Compute SSIM between the two images
        for idx, img in enumerate(grid_row):
            # Convert [-1,1] array to [0,255] int array.
            img = tf.squeeze(img)
            img = img.numpy()
            img = ((img * 0.5 + 0.5) * 255).astype("uint8")
            axs[y_counter][idx].imshow(img, cmap='gray', vmin=np.min(img), vmax=np.max(img))
            if idx == 2:
                color = "green" if is_counterfactual[
                    0] else "red"  # TODO: If multiple generators, change to count generators and not 0
                rect = Rectangle((0, 0), *img.shape[:2], fill=False, edgecolor=color, linewidth=4)
                axs[y_counter][idx].add_patch(rect)
            axs[y_counter][idx].axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0.2)
    plt.show()
    print("Hey")


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
                             training=False, num_images=6):
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
            plot_img_grid(original_images, translated_images, attention_maps, is_counterfactual_list)
            original_images = []
            translated_images = []
            attention_maps = []
            is_counterfactual_list = []
            if batch_i > 12:  # Only make 2 plots...
                break
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
