import os

from config import *
import cv2 as cv
import torch.nn as nn
import numpy as np
import torch


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def adjust_learning_rate(optimizer, shrink_factor):
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


class ExpoAverageMeter(object):
    # Exponential Weighted Average Meter
    def __init__(self, beta=0.9):
        self.reset()

    def reset(self):
        self.beta = 0.9
        self.val = 0
        self.avg = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.avg = self.beta * self.avg + (1 - self.beta) * self.val




def save_checkpoint(epoch, model, optimizer, val_loss, is_best):
    ensure_folder(save_folder)
    state = {'model': model,
             'optimizer': optimizer}
    filename = '{0}/checkpoint_{1}_{2:.3f}.tar'.format(save_folder, epoch, val_loss)
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, '{}/BEST_checkpoint.tar'.format(save_folder))


import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from sklearn.model_selection import train_test_split


def load_dataset(DATASET_DIR='./data/', image_height=360, image_width=480, BATCH_SIZE=22):
    train_dir = DATASET_DIR + 'train/'
    test_dir = DATASET_DIR + 'test/'
    train_transforms = transforms.Compose([transforms.RandomRotation(10),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])

    TRAIN_SET = dset.ImageFolder(root=train_dir, transform=train_transforms)
    TEST_SET = dset.ImageFolder(root=test_dir, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(TRAIN_SET, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(TEST_SET, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    return TRAIN_SET, TEST_SET, train_loader, test_loader


def get_rgb_value():
    f = open('./Dataset/class_dict.csv', 'r')
    lines = f.readlines()
    f.close()
    label_rgb = {}
    for idx, line in enumerate(lines[1:]):
        r, g, b = line.split(',')
        label_rgb[idx] = torch.tensor([int(r), int(g), int(b)])
    return label_rgb

def fill_seg_image(seg_image, pred_label, label_rgb):
    C, H, W = seg_image.shape
    for i in range(H):
        for j in range(W):
            seg_image[:, i, j] = label_rgb[int(pred_label[i, j])]

def get_image(result):
    # result in shape of (batchsize, 12, 360, 480)

    label_rgb = get_rgb_value()
    num_of_samples, channel, height, width = result.size()

    #seg_image = np.empty((num_of_samples, 3, height, width), dtype=np.uint8)

    seg_images = torch.zeros((num_of_samples, 3, 360, 480))

    # s = nn.Softmax(dim=1)
    # result = s(result)
    pred_labels = torch.argmax(result, dim=1)
    for idx in range(num_of_samples):
        fill_seg_image(seg_images[idx], pred_labels[idx], label_rgb)

    for i in range(num_of_samples):
        out_ = seg_images[i]
        out_ = out_.cpu().detach().numpy()
        out_ = np.transpose(out_, (1, 2, 0))
        out_ = out_ * 255
        out_ = np.clip(out_, 0, 255)
        out_ = out_.astype(np.uint8)
        out_ = cv.cvtColor(out_, cv.COLOR_RGB2BGR)
        cv.imwrite(f'{i}_out.png', out_)