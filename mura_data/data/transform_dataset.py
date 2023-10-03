import tensorflow as tf
import numpy as np
import cv2 as cv
import os
from tqdm import tqdm

from mura_data.data.load_mura import filenames
from mura_data.data.transforms import GrayScale, MinMaxNormalization, ToTensor
from mura_data.data.transforms import AdaptiveHistogramEqualization
from global_config import ROOT_DIR


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class MURASubset:
    def __init__(self, filenames, transform=None, n_channels=1, true_labels=None, patients=None):
        self.transform = transform
        self.filenames = list(filenames)
        self.n_channels = n_channels
        self.true_labels = true_labels
        self.patients = patients

    def __len__(self):
        return len(self.filenames)

    def generator(self):
        for index in range(len(self.filenames)):
            image = cv.imread(self.filenames[index])
            label = self.true_labels[index] if self.true_labels is not None else None
            patient = self.patients[index] if self.true_labels is not None else None
            filenames = self.filenames[index]

            sample = {'image': image, 'label': label, 'patient': patient, 'filename': filenames}

            if self.transform:
                sample = self.transform(sample)

            yield sample

    def to_tf_dataset(self):
        return tf.data.Dataset.from_generator(self.generator,
                                              output_signature={
                                                  'image': tf.TensorSpec(shape=(None, None, None), dtype=tf.uint8),
                                                  'label': tf.TensorSpec(shape=(), dtype=tf.int32, optional=True),
                                                  'patient': tf.TensorSpec(shape=(), dtype=tf.int32, optional=True),
                                                  'filename': tf.TensorSpec(shape=(), dtype=tf.string)
                                              })

    def __getitem__(self, index):
        image = cv.imread(self.filenames[index])
        label = self.true_labels[index] if self.true_labels is not None else None
        patient = self.patients[index] if self.true_labels is not None else None
        filenames = self.filenames[index]

        sample = {'image': image, 'label': label, 'patient': patient, 'filename': filenames}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def save_transformed_samples(self, save_dir):
        for filename in tqdm(self.filenames, total=len(self.filenames)):
            img = cv.imread(filename)
            if img is None:
                print(f"Couldn't load {filename}")
                continue

            sample = {'image': img, 'label': None, 'patient': None}
            sample = self.transform(sample)
            img = np.squeeze(sample['image'].numpy(), axis=0)

            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB) * 255.0
            img = img.astype(np.uint8)

            sample = {'image': img, 'label': None, 'patient': None}
            sample = self.transform(sample)
            img = np.squeeze(sample['image'].numpy(), axis=0)
            full_img_dir = os.path.join(save_dir, "MURA-v1.1_transformed" + filename.split("MURA-v1.1")[-1])
            file_folder = os.path.dirname(full_img_dir)
            os.makedirs(file_folder, exist_ok=True)
            cv.imwrite(full_img_dir, img * 255.0)


def apply_and_save_hist_equalization(root_dir):
    normalisation = (0, 1)
    # Assuming Compose has been refactored to support TensorFlow
    composed_transforms = Compose([
        GrayScale(),
        AdaptiveHistogramEqualization(active=True),
        MinMaxNormalization(v_min=normalisation[0], v_max=normalisation[1]),
        ToTensor()
    ])

    train_filenames, _ = filenames(root_dir, ["XR_WRIST"])
    valid_filenames, _ = filenames(root_dir, ["XR_WRIST"], train=False)
    train = MURASubset(filenames=train_filenames, transform=composed_transforms)
    validation = MURASubset(filenames=valid_filenames, transform=composed_transforms)

    train.save_transformed_samples(save_dir=root_dir)
    validation.save_transformed_samples(save_dir=root_dir)


if __name__ == '__main__':
    dataset_root = os.path.join(ROOT_DIR, "../tensorflow_datasets/")
    apply_and_save_hist_equalization(dataset_root)
