import torchvision.models
from models import segnetDown2
from models import segnetDown3
from models import segnetUp2
from models import segnetUp3
import torch.nn as nn

Vgg16 = torchvision.models.vgg16(pretrained=True)
max_pool = [4, 9, 16, 23, 30]
for param in Vgg16.features.parameters():
    param.requires_grad = False
for i in max_pool:
    Vgg16.features[i].return_indices = True

class MySegDown1(nn.Module):
    def __init__(self):
        super(MySegDown1, self).__init__()
        self.convs = Vgg16.features[0:4]
        self.max_pool = Vgg16.features[4]

    def forward(self, input):
        output = self.convs(input)
        unpooled_shape = output.size()
        output, indices = self.max_pool(output)
        return output, indices, unpooled_shape


class MySegDown2(nn.Module):
    def __init__(self):
        super(MySegDown2, self).__init__()
        self.convs = Vgg16.features[5:9]
        self.max_pool = Vgg16.features[9]

    def forward(self, input):
        output = self.convs(input)
        unpooled_shape = output.size()
        output, indices = self.max_pool(output)
        return output, indices, unpooled_shape


class MySegDown3(nn.Module):
    def __init__(self):
        super(MySegDown3, self).__init__()
        self.convs = Vgg16.features[10:16]
        self.max_pool = Vgg16.features[16]

    def forward(self, input):
        output = self.convs(input)
        unpooled_shape = output.size()
        output, indices = self.max_pool(output)
        return output, indices, unpooled_shape


class MySegDown4(nn.Module):
    def __init__(self):
        super(MySegDown4, self).__init__()
        self.convs = Vgg16.features[17:23]
        self.max_pool = Vgg16.features[23]

    def forward(self, input):
        output = self.convs(input)
        unpooled_shape = output.size()
        output, indices = self.max_pool(output)
        return output, indices, unpooled_shape


class MySegDown5(nn.Module):
    def __init__(self):
        super(MySegDown5, self).__init__()
        self.convs = Vgg16.features[24:30]
        self.max_pool = Vgg16.features[30]

    def forward(self, input):
        output = self.convs(input)
        unpooled_shape = output.size()
        output, indices = self.max_pool(output)
        return output, indices, unpooled_shape

class MYSegNet(nn.Module):
    def __init__(self, n_classes=3, in_channels=3, is_unpooling=True):
        super(MYSegNet, self).__init__()

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.down1 = MySegDown1()
        self.down2 = MySegDown2()
        self.down3 = MySegDown3()
        self.down4 = MySegDown4()
        self.down5 = MySegDown5()
        self.up5 = segnetUp3(512, 512)
        self.up4 = segnetUp3(512, 256)
        self.up3 = segnetUp3(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, n_classes)

    def forward(self, inputs):
        down1, indices_1, unpool_shape1 = self.down1.forward(inputs)
        down2, indices_2, unpool_shape2 = self.down2.forward(down1)
        down3, indices_3, unpool_shape3 = self.down3.forward(down2)
        down4, indices_4, unpool_shape4 = self.down4.forward(down3)
        down5, indices_5, unpool_shape5 = self.down5.forward(down4)

        up5 = self.up5(down5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)

        return up1