# Mura dataset transformation and loading

## Transforming with histogram equalization

These transformation steps are adapted from https://github.com/Valentyn1997/xray and require
torch and torchvision.
We use

- pytorch 1.10.2
- torchvision 0.11.3

Set the correct dataset dir and run the following command to create the transformed dataset:

```
python3 mura_data/data/transform_dataset.py
```

These images are used in the paper for both, the classifier and CycleGAN experiments.
Since all models require the same images, we did this preprocessig step only once and saved
the images to the original source data folder with the _transformed suffix.
The new folder is on the same level as MURA-v1.1 and is called MURA-v1.1-transformed.

## Loading the dataset

The dataset can now be loaded with by functions in

```
load_mura.py
```