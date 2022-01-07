import cv2 as cv
import numpy as np
from config import *
from torch.utils.data import Dataset

def get_dirs(txt_file):
    f = open(txt_file, 'r')
    lines = f.readlines()
    train, label = [], []
    for line in lines:
        train.append(line.split(' ')[0])
        label.append(line.split(' ')[1])
    f.close()
    return train, label

def load_data(dataset):
    train_files, label_files = get_dirs(dataset)
    assert len(train_files) == len(label_files)

    num_samples = len(train_files)

    x = np.empty((num_samples, 3, height, width), dtype=np.float32)
    y = np.empty((num_samples, 12, height, width), dtype=np.int64)

    for i, filename in enumerate(train_files):
        bgr_img = cv.imread(filename)
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        rgb_img = np.transpose(rgb_img, (2, 0, 1))
        assert rgb_img.shape == (3, height, width)
        assert np.max(rgb_img) <= 255
        x[i, :, :, :] = rgb_img / 255.

    for i, filename in enumerate(label_files):
        bgr_img = cv.imread(filename.strip())
        gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
        for j in range(height):
            for k in range(width):
                y[i, gray_img[j, k], j, k] = 1
    return x[:,:,:352,:], y[:,:,:352,:]

class VaeDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.x, self.y = load_data(dataset)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)