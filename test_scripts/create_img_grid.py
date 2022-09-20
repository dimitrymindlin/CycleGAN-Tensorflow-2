import matplotlib.pyplot as plt
import glob
import math

path = '/Users/dimitrymindlin/UniProjects/CycleGAN-Tensorflow-2/comparison_images/*'
folder_model_names = glob.glob(path)
#filenames = glob.glob(path)

# number of columns (width) and rows (height)
w = 2
h = 5

# create places for images
fig, axs = plt.subplots(h, w, figsize=(8, 6))

# remove axis for every image
for row in axs:
    for ax in row:
        ax.axis('off')

# display image
for model_name in folder_model_names:
    img_names = glob.glob(model_name + "/Abnormal/*")
    for i, name in enumerate(img_names):
        # calculate position
        col = i % w
        row = i // w # TODO: CHeck rows and cols!!!
        if row < h:
            # read image
            img = plt.imread(name)

            # display image
            axs[row][col].imshow(img)

            # remove axis
            # axs[row][col].axis('off')

            # add title with filename without directory
            name = name.split('/')[-1]  # keep only filename without directory
            if row == 0:
                axs[row][col].set_title(model_name.split("/")[-1])
        else:
            continue

# save in file - it has to be before `show()`
plt.savefig('grid_img.jpg')

# display all
plt.show()