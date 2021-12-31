import time

import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from models import SegNet
from utils import *

from imageio import imread, imsave

import numpy as np
import cv2 as cv

device = 'cpu'
checkpoint = './models/BEST_checkpoint.tar'  # model checkpoint
print('checkpoint: ' + str(checkpoint))

# Load models
checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
model = checkpoint['model']
model = model.to(device)
model.eval()

test_path = './images/input'
ensure_folder('./images/input')
test_images = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith('.png')].sort()
num_test_samples = len(test_images)

imgs = torch.zeros([num_test_samples, 3, imsize, imsize], dtype=torch.float, device=device)

for i, path in enumerate(test_images):
    # Read images
    img = cv.imread(path)
    img = cv.resize(img, (imsize, imsize))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # imsave('images/input/{}_image.png'.format(i), img)

    img = img.transpose(2, 0, 1)
    # img = cv.imread(path)
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # img = np.transpose(img, (2, 0, 1))
    assert img.shape == (3, imsize, imsize)
    assert np.max(img) <= 255
    img = torch.FloatTensor(img / 255.)
    imgs[i] = img


imgs = imgs.clone().detach().requires_grad_(True)

with torch.no_grad():
    preds = model(imgs)

for i in range(num_test_samples):
    out = preds[i]
    out = out.cpu().numpy()
    out = np.transpose(out, (1, 2, 0))
    out = out * 255
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)
    out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
    cv.imwrite('images/output/test{}_out.png'.format(i), out)


down1, indices_1, unpool_shape1 = model.down1(imgs)
down2, indices_2, unpool_shape2 = model.down2(down1)
down3, indices_3, unpool_shape3 = model.down3(down2)
down4, indices_4, unpool_shape4 = model.down4(down3)
down5, indices_5, unpool_shape5 = model.down5(down4)

# down5[5] = down5[5]/2+down5[6]/2
# down5[6] = down5[5]/2+down5[6]/2

for i in range(num_test_samples):
    indices_5[i] = indices_5[5]
    indices_4[i] = indices_4[5]
    indices_3[i] = indices_3[5]
    indices_2[i] = indices_2[5]
    indices_1[i] = indices_1[5]


up5 = model.up5(down5, indices_5, unpool_shape5)
up4 = model.up4(up5, indices_4, unpool_shape4)
up3 = model.up3(up4, indices_3, unpool_shape3)
up2 = model.up2(up3, indices_2, unpool_shape2)
up1 = model.up1(up2, indices_1, unpool_shape1)

for i in range(num_test_samples):
    out_ = up1[i]
    out_ = out_.cpu().detach().numpy()
    out_ = np.transpose(out_, (1, 2, 0))
    out_ = out_ * 255
    out_ = np.clip(out_, 0, 255)
    out_ = out_.astype(np.uint8)
    out_ = cv.cvtColor(out_, cv.COLOR_RGB2BGR)
    cv.imwrite(f'images/changed_feature/changed_feature_{i}_out.png', out_)
#   cv.imwrite(f'images/mixed_feature/mixed_feature_{i}_out.png', out_)

# down5 = torch.zeros(num_test_samples, 512, 7, 7)
#
# up5 = model.up5(down5, indices_5, unpool_shape5)
# up4 = model.up4(up5, indices_4, unpool_shape4)
# up3 = model.up3(up4, indices_3, unpool_shape3)
# up2 = model.up2(up3, indices_2, unpool_shape2)
# up1 = model.up1(up2, indices_1, unpool_shape1)
#
# for i in range(num_test_samples):
#     out_ = up1[i]
#     out_ = out_.cpu().detach().numpy()
#     out_ = np.transpose(out_, (1, 2, 0))
#     out_ = out_ * 255
#     out_ = np.clip(out_, 0, 255)
#     out_ = out_.astype(np.uint8)
#     out_ = cv.cvtColor(out_, cv.COLOR_RGB2BGR)
#     cv.imwrite(f'images/zero_feature/zero_feature_{i}_out.png', out_)
#
# down5 = torch.randn(num_test_samples,512,7,7)
#
# up5 = model.up5(down5, indices_5, unpool_shape5)
# up4 = model.up4(up5, indices_4, unpool_shape4)
# up3 = model.up3(up4, indices_3, unpool_shape3)
# up2 = model.up2(up3, indices_2, unpool_shape2)
# up1 = model.up1(up2, indices_1, unpool_shape1)
#
# for i in range(num_test_samples):
#     out_ = up1[i]
#     out_ = out_.cpu().detach().numpy()
#     out_ = np.transpose(out_, (1, 2, 0))
#     out_ = out_ * 255
#     out_ = np.clip(out_, 0, 255)
#     out_ = out_.astype(np.uint8)
#     out_ = cv.cvtColor(out_, cv.COLOR_RGB2BGR)
#     cv.imwrite(f'images/rand_feature/rand_feature_{i}_out.png', out_)