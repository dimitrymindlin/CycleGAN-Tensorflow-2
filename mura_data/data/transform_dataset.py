from typing import List
import sys
from torchvision.transforms import Compose
from mura_data.data import MURASubset
from mura_data.data.transforms import GrayScale, MinMaxNormalization, ToTensor
from mura_data.data.transforms import AdaptiveHistogramEqualization
from global_config import ROOT_DIR


def filenames(parts: List[str], root_dir, train=True):
    if train:
        csv_path = root_dir + "MURA-v1.1/train_image_paths.csv"
    else:
        csv_path = root_dir + "MURA-v1.1/valid_image_paths.csv"

    with open(csv_path, 'rb') as F:
        d = F.readlines()
        imgs = [root_dir + str(x, encoding='utf-8').strip() for x in d if
                str(x, encoding='utf-8').strip().split('/')[2] in parts]

    # imgs= [x.replace("/", "\\") for x in imgs]
    labels = [x.split('_')[-1].split('/')[0] for x in imgs]
    return imgs, labels


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
