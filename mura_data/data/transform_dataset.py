from typing import List
import sys
from torchvision.transforms import Compose
from mura_data.data import MURASubset
from mura_data.data.load_mura import filenames
from mura_data.data.transforms import GrayScale, MinMaxNormalization, ToTensor
from mura_data.data.transforms import AdaptiveHistogramEqualization
from global_config import ROOT_DIR


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
