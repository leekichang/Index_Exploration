import models
import torchvision.models
from utils import *
import cv2 as cv
import os
import numpy as np

checkpoint = './models/BEST_checkpoint.tar'  # model checkpoint
checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
SegNet = checkpoint['model']
SegNet = SegNet.to(device)

Vgg16 = torchvision.models.vgg16(pretrained=True)
max_pool = [4, 9, 16, 23, 30]
for i in max_pool:
    Vgg16.features[i].return_indices=True

TRAIN_SET, TEST_SET, train_loader, test_loader = load_dataset()

Vgg16.eval()

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
    # imsave('images/input/{}_image.png'.format(i), img)

    img = img.transpose(2, 0, 1)
    # img = cv.imread(path)
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # img = np.transpose(img, (2, 0, 1))
    assert img.shape == (3, imsize, imsize)
    assert np.max(img) <= 255
    img = torch.FloatTensor(img / 255.)
    imgs[i] = img


temp = imgs.clone().detach().requires_grad_(True)
vgg_indices_1 = torch.zeros((9, 64, 112, 112))
vgg_indices_2 = torch.zeros((9, 128, 56, 56))
vgg_indices_3 = torch.zeros((9, 256, 28, 28))
vgg_indices_4 = torch.zeros((9, 512, 14, 14))
vgg_indices_5 = torch.zeros((9, 512, 7, 7))

indices = [vgg_indices_1, vgg_indices_2, vgg_indices_3, vgg_indices_4, vgg_indices_5]

count = 0

with torch.no_grad():
    for i in range(len(Vgg16.features)):
        if i not in max_pool:
            temp = Vgg16.features[i](temp)
        else:
            temp, indices[count] = Vgg16.features[i](temp)
            count += 1

down1, indices_1, unpool_shape1 = SegNet.down1(imgs)
down2, indices_2, unpool_shape2 = SegNet.down2(down1)
down3, indices_3, unpool_shape3 = SegNet.down3(down2)
down4, indices_4, unpool_shape4 = SegNet.down4(down3)
down5, indices_5, unpool_shape5 = SegNet.down5(down4)
# up5 = SegNet.up5(down5, indices_5, unpool_shape5)
# up4 = SegNet.up4(up5, indices_4, unpool_shape4)
# up3 = SegNet.up3(up4, indices_3, unpool_shape3)
# up2 = SegNet.up2(up3, indices_2, unpool_shape2)
# up1 = SegNet.up1(up2, indices_1, unpool_shape1)

up5 = SegNet.up5(down5, indices[4], unpool_shape5)
up4 = SegNet.up4(up5, indices[3], unpool_shape4)
up3 = SegNet.up3(up4, indices[2], unpool_shape3)
up2 = SegNet.up2(up3, indices[1], unpool_shape2)
up1 = SegNet.up1(up2, indices[0], unpool_shape1)

for i in range(num_test_samples):
    out = up1[i]
    out = out.cpu().detach().numpy()
    out = np.transpose(out, (1, 2, 0))
    out = out * 255
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)
    out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
    cv.imwrite('images/vgg_output/vgg{}_out.png'.format(i), out)

# print(torch.tensor(np.array(indices)).shape, indices_5.shape)

