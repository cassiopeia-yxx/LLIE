# -*- coding: utf-8 -*-
# Auther : ML_XX
# Date : 2023/12/19 20:54
# File : network.py

import torch
import torch.nn as nn
from MCSC.model import MCSC
import cv2
import numpy as np

class DeNet(nn.Module):
    def __init__(self, args):
        super(DeNet, self).__init__()
        # self.CSC = self.make_CSC(args, CSC.CSC_UTV)
        self.MCSC = MCSC.MCSCNet(args.in_channel, num_filters=32)
        # self.iter = args.iter



    def make_CSC(self, args, CSC_UTV):
        layers = []
        for i in range(args.iter):
            layers.append(CSC_UTV(args.DCU_T, 1, 32, 3))
        return nn.Sequential(*layers)


    def make_MCSC(self, args, MCSCNet):
        layers = []
        for i in range(args.iter):
            layers.append(MCSCNet(args.in_channel, num_filters=32))
        return nn.Sequential(*layers)

    def forward(self, input):
        batch, channel, h, w = input.size()
        zeros = torch.zeros(batch, channel, h, w).cuda()
        R, scale_1, scale_2, scale_3 = self.MCSC(input)
        # R = torch.min(R, zeros)
        # scale_1 = torch.min(scale_1, zeros)
        # scale_2 = torch.min(scale_2, zeros)
        # scale_3 = torch.min(scale_3, zeros)
        return R, scale_1, scale_2, scale_3
