import glob
import re
import pandas as pd
from numpy import uint8
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import imutils
import os
import sys
from glob import glob
from matplotlib import pyplot as plt
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
