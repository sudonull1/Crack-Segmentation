#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 17:09:49 2021

@author: sudonull
"""
"""pytorch implementation of UNet on ResNet Netwroks"""
import torch
import torch.nn as nn
from torchsummary import summary
from torch.nn import functional as F
import torchvision

""" 
    double convolution --> (conv2d->BN->LeakyReLU->conv2d->BN->LeakyReLU)
    
    using leakyReLU decreases training time 
    LeakyReLU --> neg_slope = 0.01(default), inplace = True
    
    encoder block
"""

BatchNorm2d = nn.SyncBatchNorm
class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel, momentum = 0.1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channel,out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel, momentum = 0.1),
            nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        return self.encoder(x)

"""Conv-Relu Activation"""

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

"""simple implimentation of encoder"""
class Double_Conv(nn.Module):
    def __init__(self, in_, middle, out):
        super(Double_Conv, self).__init__()
        self.in_ = in_
        self.block = nn.Sequential(
            ConvRelu(in_, middle),
            nn.ConvTranspose2d(middle, out, kernel_size=4, stride=2, padding=1), 
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

"""DownScaling with maxpool then DoubleConv"""
class Down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.max_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channel, out_channel)
            )
    def forward(self, x):
        return self.max_conv(x)

"""
    upconvolution
    1*1 ==> 2*2

"""
def up_conv(in_, out):
    return nn.ConvTranspose2d(in_, out, kernel_size=2, stride=2)


"""Unet with Deep ResNet encoders"""

class DeepUnetResNet(nn.Module):
    def __init__(self, encoder_depth, num_classes, num_filters=32, dropout_2d=0.2,
                 pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d
        if encoder_depth == 34:
            channel=1024
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        else:
            channel=2048
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.LeakyReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = Double_Conv(channel, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = Double_Conv(channel + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = Double_Conv(channel // 2 + num_filters * 8, num_filters * 8 * 2, num_filters *8)
        self.dec3 = Double_Conv(channel// 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = Double_Conv(channel// 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2*2)
        self.dec1 = Double_Conv(num_filters * 2 * 2, num_filters * 2 * 2, num_filters)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        pool = self.pool(conv5)
        center = self.center(pool)

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        return self.final(F.dropout2d(dec0, p=self.dropout_2d))


if __name__ == "__main__":
    z = torch.randn(1, 3,512,512)
    encod = torchvision.models.resnet101(pretrained=True)
    model = DeepUnetResNet(encoder_depth = 101, num_classes=1, num_filters=32, pretrained=True)
    print(model)
    preds=model(z)
    print(preds.shape)
    print(z.shape)






