import logging
import math
import sys

import torch
import torch.nn as nn

from MobileNetV2 import MobileNetV2, InvertedResidual
def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )
class Attention_block(nn.Module):   #with attention 
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class MobileUNet(nn.Module):
    def __init__(self, class_num = 1, pre_train = '../model/mobilenet_v2.pth.tar'):
        super(MobileUNet, self).__init__()
        self.UpDecodeNet = []
        self.class_num = class_num
        self.base_model = MobileNetV2()
        self.dconv1 = nn.ConvTranspose2d(1280, 96, kernel_size = 2, stride = 2)
        self.invres1 = InvertedResidual(192, 96, 1, 6)

        self.dconv2 = nn.ConvTranspose2d(96, 32, kernel_size = 2,  stride = 2)
        self.invres2 = InvertedResidual(64, 32, 1, 6)

        self.dconv3 = nn.ConvTranspose2d(32, 24, kernel_size = 2, stride = 2)
        self.invres3 = InvertedResidual(48, 24, 1, 6)

        self.dconv4 = nn.ConvTranspose2d(24, 16, kernel_size = 2, stride = 2)
        self.invres4 = InvertedResidual(32, 16, 1, 6)

        self.conv_last = nn.Conv2d(16, 3, kernel_size = 1)
        self.conv_score = nn.Conv2d(3, self.class_num, 1)

        self.att1 = Attention_block(F_g=16,F_l=16,F_int=8)
        self.att2 = Attention_block(F_g=24,F_l=24,F_int=12)
        self.att3 = Attention_block(F_g=32,F_l=32,F_int=16)
        self.att4 = Attention_block(F_g=96,F_l=96,F_int=48)

        self._init_weights()

        if pre_train is not None:
            print("pre_train.........")
            self.base_model.load_state_dict(torch.load(pre_train))


    def forward(self, input):
        outsize = input.size()[2:]
        layer = input
        for index in range(0, 2):
            layer = self.base_model.features[index](layer)
        concat1 = layer
        for index in range(2, 4):
            layer = self.base_model.features[index](layer)
        concat2 = layer
        for index in range(4, 7):
            layer = self.base_model.features[index](layer)
        concat3 = layer
        for index in range(7, 14):
            layer = self.base_model.features[index](layer)
        concat4 = layer
        for index in range(14, 19):
            layer = self.base_model.features[index](layer)
        d1 = self.dconv1(layer)
        concat4 = self.att4(g=d1, x=concat4)
        up1 = torch.cat([concat4, d1], dim = 1)

        up1 = self.invres1(up1)
        d2 = self.dconv2(up1)
        concat3 = self.att3(g=d2, x=concat3)
        up2 = torch.cat([concat3, d2], dim = 1)

        up2 = self.invres2(up2)
        d3 = self.dconv3(up2)
        concat2 = self.att2(g=d3, x=concat2)
        up3 = torch.cat([concat2,d3], dim = 1)

        up3 = self.invres3(up3)
        d4 = self.dconv4(up3)
        concat1 = self.att1(g=d4, x=concat1)
        up4 = torch.cat([concat1, d4], dim = 1)

        up4 = self.invres4(up4)
        layer = self.conv_last(up4)
        layer = self.conv_score(layer)
        layer = torch.nn.functional.upsample_bilinear(layer, outsize)

        # layer = torch.nn.Sigmoid()(layer)
        # print(layer.shape,'layer')
        return layer

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()