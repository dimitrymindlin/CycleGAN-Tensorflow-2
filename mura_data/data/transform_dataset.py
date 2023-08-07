from typing import List
import sys
from torchvision.transforms import Compose
from mura_data.data.load_mura import filenames
from mura_data.data.transforms import GrayScale, MinMaxNormalization, ToTensor
from mura_data.data.transforms import AdaptiveHistogramEqualization
from global_config import ROOT_DIR

from numpy import uint8

from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import os
from tqdm import tqdm


class MURASubset(Dataset):

    def __init__(self, filenames, transform=None, n_channels=1, true_labels=None, patients=None):
        """Initialization
        :param filenames: list of filenames, e.g. from TrainValTestSplitter
        :param true_labels: list of true labels (for validation and split)
        """
        self.transform = transform
        self.filenames = list(filenames)
        self.n_channels = n_channels
        self.true_labels = true_labels
        self.patients = patients

    def __len__(self) -> int:
        """Denotes the number of batches per epoch"""
        return len(self.filenames)

    def __getitem__(self, index) -> np.array:
        """Reads sample"""
        image = cv.imread(self.filenames[index])
        label = self.true_labels[index] if self.true_labels is not None else None
        patient = self.patients[index] if self.true_labels is not None else None
        filenames = self.filenames[index]

        sample = {'image': image, 'label': label, 'patient': patient,
                  'filename': filenames}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def save_transformed_samples(self, save_dir):
        for filename in tqdm(self.filenames, maxinterval=len(self.filenames)):
            img = cv.imread(filename)
            if img is None:
                print(f"Couldn't load {filename}")
                continue

            sample = {'image': img, 'label': None, 'patient': None}
            sample = self.transform(sample)
            img = np.squeeze(sample['image'].numpy(), axis=0)

            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB) * 255.0
            img = img.astype(uint8)

            sample = {'image': img, 'label': None, 'patient': None}
            sample = self.transform(sample)
            img = np.squeeze(sample['image'].numpy(), axis=0)
            full_img_dir = save_dir + "MURA-v1.1_transformed" + filename.split("MURA-v1.1")[-1]
            file_folder = "/".join(full_img_dir.split("/")[:-1])
            try:
                os.makedirs(file_folder)
            except FileExistsError:
                pass
            cv.imwrite(full_img_dir, img * 255.0)


def apply_and_save_hist_equalization(root_dir):
    normalisation = (0, 1)
    composed_transforms = Compose([GrayScale(),
                                   AdaptiveHistogramEqualization(active=True),
                                   MinMaxNormalization(v_min=normalisation[0], v_max=normalisation[1]),
                                   ToTensor()])

    train_filenames, _ = filenames(["XR_WRIST"], root_dir)
    valid_filenamse, _ = filenames(["XR_WRIST"], root_dir, train=False)
    train = MURASubset(filenames=train_filenames, transform=composed_transforms)
    validation = MURASubset(filenames=valid_filenamse, transform=composed_transforms)

    train.save_transformed_samples(save_dir=root_dir)
    validation.save_transformed_samples(save_dir=root_dir)


if __name__ == '__main__':
    # get root dir
    dataset_root = ROOT_DIR + "/../tensorflow_datasets/"
    apply_and_save_hist_equalization(dataset_root)
