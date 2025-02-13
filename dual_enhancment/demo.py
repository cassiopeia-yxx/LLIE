# -*- coding: utf-8 -*-
# Auther : ML_XX
# Date : 2024/1/11 12:57
# File : demo.py
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import PWAS
from uilts import *
import pyiqa
from pyiqa import imread2tensor
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


class Parameter:
    def __init__(self):
        # iters
        self.outer_iter = 100
        # update S paramete1s
        self.lambda3 = 0.1
        # update B parameters
        self.mu = 0.1
        # stop condition
        self.tol = 1e-6

image = cv2.imread('1.bmp')
parameter = Parameter()
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
v_channel = hsv_image[:, :, 2].astype(np.float32)
input_image = np.log(0.001 + v_channel)
R, L, err_L, err_R, err_B = PWAS.PWAS(input_image, parameter)
R1 = np.exp(R)
L1 = np.exp(L)
enhanceV = gamma_correct(2.2, R1, L1)
hsv_image[:, :, 2] = enhanceV
enhanced_result = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
cv2.imwrite('enhanced_result.bmp', enhanced_result)
