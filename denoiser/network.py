# -*- coding: utf-8 -*-
# Auther : ML_XX
# Date : 2024/1/9 14:20
# File : network.py
from torch import nn
import torch

from denoiser import basicblock


class IRCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64):
        super(IRCNN, self).__init__()
        L =[]
        L.append(nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, stride=1, padding=1,bias=False))
        L.append(nn.LeakyReLU(0.02))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False))
        L.append(nn.LeakyReLU(0.02))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False))
        L.append(nn.LeakyReLU(0.02))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False))
        L.append(nn.LeakyReLU(0.02))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False))
        L.append(nn.LeakyReLU(0.02))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False))
        L.append(nn.LeakyReLU(0.02))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False))
        L.append(nn.LeakyReLU(0.02))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False))
        L.append(nn.LeakyReLU(0.02))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False))
        L.append(nn.LeakyReLU(0.02))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False))
        L.append(nn.LeakyReLU(0.02))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False))
        L.append(nn.LeakyReLU(0.02))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False))
        L.append(nn.LeakyReLU(0.02))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False))
        L.append(nn.LeakyReLU(0.02))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False))
        L.append(nn.LeakyReLU(0.02))
        L.append(nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, bias=False))
        self.model = basicblock.sequential(*L)

    def forward(self, x):
        n = self.model(x)
        return x - n