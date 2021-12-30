import os

import cv2 as cv
import numpy as np
from torch.utils.data import Dataset

from config import imsize


def load_data(split):
    # (num_samples, 320, 320, 4)
    num_samples = 20580
    train_split = 0.8
    num_train = int(round(num_samples * train_split))
    num_valid = num_samples - num_train
    num_mix = 2
    
    if split == 'train':
        num_samples = num_train
        folder = '../data/train_ae/doggy'
    elif split == 'mix':
        num_samples = num_mix
        folder = './images/'
    else:
        num_samples = num_train
        folder = '../data/test'

    x = np.empty((num_samples, 3, imsize, imsize), dtype=np.float32)
    y = np.empty((num_samples, 3, imsize, imsize), dtype=np.float32)
    
    files = [os.path.join(folder, f) for f in os.listdir(folder)]
    #print(files)
    if split == 'train':
        for i, filename in enumerate(files[:num_samples]):
            bgr_img = cv.imread(filename)
            rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
            rgb_img = np.transpose(rgb_img, (2, 0, 1))
            # print('rgb_img.shape: ' + str(rgb_img.shape))
            #print(rgb_img.shape)
            assert rgb_img.shape == (3, imsize, imsize)
            assert np.max(rgb_img) <= 255
            # print("BHKOO", i)
            # print(rgb_img.shape)
            x[i, :, :, :] = rgb_img / 255.
            y[i, :, :, :] = rgb_img / 255.
    else:
        for i, filename in enumerate(files[num_samples:]):
            bgr_img = cv.imread(filename)
            rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
            rgb_img = np.transpose(rgb_img, (2, 0, 1))
            # print('rgb_img.shape: ' + str(rgb_img.shape))
            #print(rgb_img.shape)
            assert rgb_img.shape == (3, imsize, imsize)
            assert np.max(rgb_img) <= 255
            # print("BHKOO", i)
            # print(rgb_img.shape)
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
