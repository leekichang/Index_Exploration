import time

import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from models import SegNet
from utils import *
from parser import *

from imageio import imread, imsave

import numpy as np
import cv2 as cv

args = parse_args()

device = args.device #'cpu'

checkpoint = args.model #'./models/BEST_checkpoint.tar'  # model checkpoint
print('checkpoint: ' + str(checkpoint))

# Load models
checkpoint = torch.load(checkpoint, map_location=torch.device(device))
model = checkpoint['model']
model = model.to(device)
model.eval()

test_path = './images/input'
ensure_folder('./images/input')
test_images = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith('.png')]
test_images.sort()
num_test_samples = len(test_images)

imgs = torch.zeros([num_test_samples, 3, imsize, imsize], dtype=torch.float, device=device)

for i, path in enumerate(test_images):
    # Read images
    img = cv.imread(path)
    img = cv.resize(img, (imsize, imsize))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    img = img.transpose(2, 0, 1)
    # img = cv.imread(path)
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # img = np.transpose(img, (2, 0, 1))
    assert img.shape == (3, imsize, imsize)
    assert np.max(img) <= 255
    img = torch.FloatTensor(img / 255.)
    imgs[i] = img

imgs = imgs.clone().detach().requires_grad_(False)

down1, indices_1, unpool_shape1 = model.down1(imgs)
down2, indices_2, unpool_shape2 = model.down2(down1)
down3, indices_3, unpool_shape3 = model.down3(down2)
down4, indices_4, unpool_shape4 = model.down4(down3)
down5, indices_5, unpool_shape5 = model.down5(down4)
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
#    cv.imwrite(f'images/{i}_out.png', out_)

# if args.type == 'output':
#     with torch.no_grad():
#         preds = model(imgs)
#
#     for i in range(num_test_samples):
#         out = preds[i]
#         out = out.cpu().numpy()
#         out = np.transpose(out, (1, 2, 0))
#         out = out * 255
#         out = np.clip(out, 0, 255)
#         out = out.astype(np.uint8)
#         out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
#         cv.imwrite('images/trained_vgg_output/test{}_out.png'.format(i), out)
# else:
#     '''
#     args.type can be 'rand', 'zero', 'trained_rand', 'trained_zero'
#     '''
#     down1, indices_1, unpool_shape1 = model.down1(imgs)
#     down2, indices_2, unpool_shape2 = model.down2(down1)
#     down3, indices_3, unpool_shape3 = model.down3(down2)
#     down4, indices_4, unpool_shape4 = model.down4(down3)
#     down5, indices_5, unpool_shape5 = model.down5(down4)
#     if args.type == 'zero' or args.type == 'trained_zero':
#         down5 = torch.zeros(down5.size())
#         file_name = 'zero_feature'
#     elif args.type == 'rand' or args.type == 'trained_rand':
#         down5 = torch.randn(down5.size())
#         file_name = 'rand_feature'
#     up5 = model.up5(down5, indices_5, unpool_shape5)
#     up4 = model.up4(up5, indices_4, unpool_shape4)
#     up3 = model.up3(up4, indices_3, unpool_shape3)
#     up2 = model.up2(up3, indices_2, unpool_shape2)
#     up1 = model.up1(up2, indices_1, unpool_shape1)
#
#     for i in range(num_test_samples):
#         out_ = up1[i]
#         out_ = out_.cpu().detach().numpy()
#         out_ = np.transpose(out_, (1, 2, 0))
#         out_ = out_ * 255
#         out_ = np.clip(out_, 0, 255)
#         out_ = out_.astype(np.uint8)
#         out_ = cv.cvtColor(out_, cv.COLOR_RGB2BGR)
#         cv.imwrite(f'images/{args.type}_feature/{file_name}_{i}_out.png', out_)











# for j in range(9):
#     temp2 = up2[j]
#     temp3 = up3[j]
#     temp4 = up4[j]
#     temp5 = up5[j]
#     for i in range(64):
#         out = temp2[i].detach().cpu().numpy()
#         #out = np.transpose(out, (1, 2, 0))
#         out = out * 255
#         out = np.clip(out, 0, 255)
#         out = out.astype(np.uint8)
#         #out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
#         cv.imwrite(f'images/IndicesMap/{j}_image/rand_up2/up2_0_{i}_out.png', out)
#     for i in range(128):
#         out = temp3[i].detach().cpu().numpy()
#         #out = np.transpose(out, (1, 2, 0))
#         out = out * 255
#         out = np.clip(out, 0, 255)
#         out = out.astype(np.uint8)
#         #out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
#         cv.imwrite(f'images/IndicesMap/{j}_image/rand_up3/up3_0_{i}_out.png', out)
#     for i in range(256):
#         out = temp4[i].detach().cpu().numpy()
#         #out = np.transpose(out, (1, 2, 0))
#         out = out * 255
#         out = np.clip(out, 0, 255)
#         out = out.astype(np.uint8)
#         #out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
#         cv.imwrite(f'images/IndicesMap/{j}_image/rand_up4/up4_0_{i}_out.png', out)
#     for i in range(512):
#         out = temp5[i].detach().cpu().numpy()
#         #out = np.transpose(out, (1, 2, 0))
#         out = out * 255
#         out = np.clip(out, 0, 255)
#         out = out.astype(np.uint8)
#         #out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
#         cv.imwrite(f'images/IndicesMap/{j}_image/rand_up5/up5_0_{i}_out.png', out)
# for j in range(9):
#     for i in range(512):
#         out = down5[j][i].detach().cpu().numpy()
#         #out = np.transpose(out, (1, 2, 0))
#         out = out * 255
#         out = np.clip(out, 0, 255)
#         out = out.astype(np.uint8)
#         #out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
#         cv.imwrite(f'images/IndicesMap/{j}_image/FeatureMap/feature_0_{i}.png', out)
