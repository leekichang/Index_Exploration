import os

from config import *


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


import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from sklearn.model_selection import train_test_split


def load_dataset(DATASET_DIR='./data/', image_height=224, BATCH_SIZE=256):
    train_dir = DATASET_DIR + 'train/'
    test_dir = DATASET_DIR + 'test/'
    train_transforms = transforms.Compose([transforms.RandomRotation(10),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomResizedCrop(image_height),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.RandomResizedCrop(image_height),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])

    TRAIN_SET = dset.ImageFolder(root=train_dir, transform=train_transforms)
    TEST_SET = dset.ImageFolder(root=test_dir, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(TRAIN_SET, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(TEST_SET, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    return TRAIN_SET, TEST_SET, train_loader, test_loader