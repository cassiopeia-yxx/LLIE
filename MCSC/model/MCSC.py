# -*- coding: utf-8 -*-
# Auther : ML_XX
# Date : 2023/12/24 16:38
# File : CSC.py

import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_, xavier_uniform_

class MCSCNet(nn.Module):
    def __init__(self, channel, num_filters):
        super(MCSCNet, self).__init__()
        self.channel = channel
        self.num_filters = num_filters

        self.scale_1 = MCSC(self.channel, self.num_filters, 7)
        padding = int(int(1+(7-1))//2)
        self.conv_1 = nn.Conv2d(self.num_filters, self.channel, 7, stride=1, padding=padding, bias=False)

        self.scale_2 = MCSC(self.channel, self.num_filters, 5)
        padding = int(int(1+(5-1))//2)
        self.conv_2 = nn.Conv2d(self.num_filters, self.channel, 5, stride=1, padding=padding, bias=False)

        self.scale_3 = MCSC(self.channel, self.num_filters, 3)
        padding = int(int(1+(3-1))//2)
        # self.conv_3 = nn.Conv2d(self.num_filters, self.channel, 3, stride=1, padding=padding, bias=False)
        #
        # self.scale_4 = MCSC(self.channel, self.num_filters, 7)
        # padding = int(int(1 + (7 - 1)) // 2)
        # self.conv_4 = nn.Conv2d(self.num_filters, self.channel, 7, stride=1, padding=padding, bias=False)
        # self.scale_5 = MCSC(self.channel, self.num_filters, 5)
        # padding = int(int(1 + (5 - 1)) // 2)
        # self.conv_5 = nn.Conv2d(self.num_filters, self.channel, 5, stride=1, padding=padding, bias=False)
        # self.scale_6 = MCSC(self.channel, self.num_filters, 3)
        self.decoder = decoder(self.channel, self.num_filters)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         xavier_uniform_(m.weight)


    def forward(self, x):
        # The first round
        z1 = self.scale_1(x)
        x1 = self.conv_1(z1)
        x2_hat = x - x1
        z2 = self.scale_2(x2_hat)
        x2 = self.conv_2(z2)
        x3_hat = x2_hat - x2
        z3 = self.scale_3(x3_hat)
        # The second round
        # x3 = self.conv_3(z3)
        # x1_hat = x - x3 - x2
        # second_z1 = self.scale_4(x1_hat)
        # second_x1 = self.conv_4(second_z1)
        # x2hat = x - second_x1
        # second_z2 = self.scale_5(x2hat)
        # second_x2 = self.conv_5(second_z2)
        # x3hat = x2hat - second_x2
        # second_z3 = self.scale_6(x3hat)

        f_pred, x1, x2, x3 = self.decoder(z1, z2, z3)
        return f_pred, x1, x2, x3


class decoder(nn.Module):
    def __init__(self, channel, filters):
        super(decoder, self).__init__()
        self.channel = channel
        self.filters = filters
        padding = int(int(1 + (7 - 1)) // 2)
        self.decoconv1 = nn.Conv2d(in_channels=self.filters, out_channels=self.channel, kernel_size=7, stride=1,
                                   padding=padding, bias=False)
        padding = int(int(1 + (5 - 1)) // 2)
        self.decoconv2 = nn.Conv2d(in_channels=self.filters, out_channels=self.channel, kernel_size=5, stride=1,
                                   padding=padding, bias=False)
        padding = int(int(1 + (3 - 1)) // 2)
        self.decoconv3 = nn.Conv2d(in_channels=self.filters, out_channels=self.channel, kernel_size=3, stride=1,
                                   padding=padding, bias=False)
    def forward(self, z1, z2, z3):
        rec_x1 = self.decoconv1(z1)
        rec_x2 = self.decoconv2(z2)
        rec_x3 = self.decoconv3(z3)
        rec_x = rec_x1 + rec_x2 + rec_x3
        return rec_x, rec_x1, rec_x2, rec_x3



class MCSC(nn.Module):
    def __init__(self, in_channel,  out_channel,  kernel_size,
                ):
        super(MCSC, self).__init__()

        padding = (kernel_size-1)//2
        layer_in = nn.Conv2d(in_channel, out_channel, kernel_size, stride=1, padding=padding, dilation=1, groups=1, bias=False)
        bn_in = nn.BatchNorm2d(out_channel)
        first_layer = [
            layer_in,
            bn_in,
            nn.LeakyReLU(0.2)
        ]
        self.first_layer = nn.Sequential(*first_layer)
        num_blocks = 10
        CSC = []
        for i in range(num_blocks):
            CSC.append(CSCNet(in_channel, out_channel, kernel_size, padding))
        self.CSC = nn.Sequential(*CSC)
        self.num_blocks = num_blocks

    def forward(self, x):
        z = self.first_layer(x)
        for i in range(self.num_blocks):
            z = self.CSC[i](x, z)
        return z


class CSCNet(nn.Module):
    def __init__(self, in_channels, out_channel, kernel_size, padding):
        super(CSCNet, self).__init__()
        self.decoder = self.conv_decoder(in_channels, out_channel, kernel_size, padding)
        self.encoder = self.conv_encoder(in_channels, out_channel, kernel_size, padding)
        self.BN = nn.BatchNorm2d(out_channel)
        self.leak_relu = nn.LeakyReLU(0.2)

    def conv_decoder(self, in_channels, out_channel, kernel_size, padding):
        decoder = nn.Conv2d(out_channel, in_channels, kernel_size, stride=1, padding=padding, bias=False)
        return decoder

    def conv_encoder(self, in_channels, out_channel, kernel_size, padding):
        encoder = nn.Conv2d(in_channels, out_channel, kernel_size, stride=1, padding=padding, bias=False)
        return encoder

    def forward(self, data_x, code_z):
        decode_z = self.decoder(code_z)
        temp = data_x - decode_z
        detemp = self.encoder(temp)
        code_z = code_z + detemp
        code_z = self.BN(code_z)
        code_z = self.leak_relu(code_z)
        return code_z

