import time

import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from models import SegNet
# from utils import *
from CamVid_utils import *
from config import *
from FCN import *
import numpy as np
import cv2 as cv


checkpoint = './models/asdBEST_checkpoint.tar'  # model checkpoint
print('checkpoint: ' + str(checkpoint))

# Load models
checkpoint = torch.load(checkpoint, map_location=torch.device(device))
model = checkpoint['model']
model = model.to(device)
model.eval()

test_path = './test/input'
ensure_folder(test_path)
test_images = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith('.png')]
test_images.sort()
num_test_samples = len(test_images)

imgs = torch.zeros([num_test_samples, 3, height, width], dtype=torch.float, device=device)

for i, path in enumerate(test_images):
    # Read images
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    img = img.transpose(2, 0, 1)

    img = torch.FloatTensor(img / 255.)
    imgs[i] = img[:, :352,:]

imgs = imgs.clone().detach().requires_grad_(False)

out = model(imgs)

get_image(out[0])

# down1, indices_1, unpool_shape1 = model.down1(imgs)
# down2, indices_2, unpool_shape2 = model.down2(down1)
# down3, indices_3, unpool_shape3 = model.down3(down2)
# down4, indices_4, unpool_shape4 = model.down4(down3)
# down5, indices_5, unpool_shape5 = model.down5(down4)
# up5 = model.up5(down5, indices_5, unpool_shape5)
# up4 = model.up4(up5, indices_4, unpool_shape4)
# up3 = model.up3(up4, indices_3, unpool_shape3)
# up2 = model.up2(up3, indices_2, unpool_shape2)
# up1 = model.up1(up2, indices_1, unpool_shape1)
# out = model.softmax(up1)
#
# get_image(out)
#mean = (down5[0]+down5[1]+down5[2]+down5[3]+down5[4])/5
#indices_1 = torch.flip(indices_1, [3])
#indices_2 = torch.flip(indices_2, [3])
#indices_3 = torch.flip(indices_3, [3])
#indices_4 = torch.flip(indices_4, [3])
#indices_5 = torch.flip(indices_5, [3])
# down5 = torch.zeros(down5.size(), device=device)
#down5[0] = mean
#down5[1] = mean
#down5[2] = mean
#down5[3] = mean
#down5[4] = mean
#print(down5.shape)
#up5 = model.up5(down5, indices_5, unpool_shape5)
#up4 = model.up4(up5, indices_4, unpool_shape4)
#up3 = model.up3(up4, indices_3, unpool_shape3)
#up2 = model.up2(up3, indices_2, unpool_shape2)
#up1 = model.up1(up2, indices_1, unpool_shape1)
#out = model.softmax(up1)
#get_image(out)