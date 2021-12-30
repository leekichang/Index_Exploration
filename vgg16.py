import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time

import numpy as np
import torch
from models import Vgg16
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import sklearn.metrics as metrics

import utils

DATASET_DIR = '../../Dog_Breed_Classification/data'

BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 20

SaveModelName = "vgg16"

ModelSavePath = "./models/" + SaveModelName + "/"
if not os.path.isdir(ModelSavePath):
    os.mkdir(ModelSavePath)

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("Working with", DEVICE)

TRAIN_SET, TEST_SET, train_loader, test_loader = utils.load_dataset()