import os

import cv2 as cv
import numpy as np
from torch.utils.data import Dataset

from utils.config import imsize


def load_data(split):
    # (num_samples, 320, 320, 4)
    num_samples = 20580
    train_split = 0.8
    num_train = int(num_samples * train_split)
    num_valid = num_samples - num_train
    num_mix = 2
    
    if split == 'train':
        num_samples = num_train
        folder = '../../Dog_Breed_Classification/data/train_ae/doggy'
    else:
        num_samples = num_valid
        folder = '../../Dog_Breed_Classification/data/train_ae/doggy'

    x = np.empty((num_samples, 3, imsize, imsize), dtype=np.float32)
    y = np.empty((num_samples, 3, imsize, imsize), dtype=np.float32)
    
    files = [os.path.join(folder, f) for f in os.listdir(folder)]
    if split == 'train':
        for i, filename in enumerate(files[:num_train]):
            bgr_img = cv.imread(filename)
            rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
            rgb_img = np.transpose(rgb_img, (2, 0, 1))
            assert rgb_img.shape == (3, imsize, imsize)
            assert np.max(rgb_img) <= 255
            x[i, :, :, :] = rgb_img / 255.
            y[i, :, :, :] = rgb_img / 255.
    else:
        for i, filename in enumerate(files[num_train:]):
            bgr_img = cv.imread(filename)
            rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
            rgb_img = np.transpose(rgb_img, (2, 0, 1))
            assert rgb_img.shape == (3, imsize, imsize)
            assert np.max(rgb_img) <= 255
            x[i, :, :, :] = rgb_img / 255.
            y[i, :, :, :] = rgb_img / 255.

    return x, y


class VaeDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.x, self.y = load_data(split)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)
