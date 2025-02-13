# -*- coding: utf-8 -*-
# Auther : ML_XX
# Date : 2024/1/12 10:19
# File : uilts.py
import math

import numba
import torch
from scipy.ndimage import convolve
from scipy.signal import convolve2d
import cv2
import numpy as np
import cupy as cp
from torch.autograd import Variable
from torch.utils.dlpack import to_dlpack,from_dlpack


def soft(x, tau):
    y = cp.sign(x) * cp.maximum(cp.abs(x) - tau, 0)
    return y


def normalize(mat):
    dim = mat.shape[0]
    sum = np.sum(mat ** 2, axis=0)
    sqrt = np.sqrt(sum)
    sqrt_eps = sqrt + np.finfo(np.float64).eps
    mat = mat / sqrt_eps
    return mat






def gamma_correct(gamma, B, S):
    L_gamma = 255 * ((S / 255) ** (1 / gamma))
    enhanceV = B * L_gamma
    enhanceV = np.uint8(enhanceV)
    return enhanceV


def update_s(O, B, lambda3):
    lambda3 = lambda3 * 2
    h, w = O.shape
    fx = np.array([[1, -1]])
    fy = np.array([[1], [-1]])
    F0B = O - B
    Fx = np.fft.fft2(fx, (h, w))
    Fy = np.fft.fft2(fy, (h, w))
    FQTQ = np.abs(Fx) ** 2 + np.abs(Fy) ** 2
    Deleft = 1 + lambda3 * (FQTQ)
    FS = np.fft.fft2(F0B) / Deleft
    S = np.real(np.fft.ifft2(FS))
    S = np.maximum(S, O)
    return S


def update_z(image, MCSCNet, opt):
    v_image = np.expand_dims(image, (0, 1))
    v_image = Variable(torch.Tensor(v_image))
    v_image = v_image.cuda()
    with torch.no_grad():
        test_output, scale_1, scale_2, scale_3 = MCSCNet(v_image)
        scale_1 = scale_1.detach().cpu().numpy().squeeze(0).squeeze(0)
        scale_2 = scale_2.detach().cpu().numpy().squeeze(0).squeeze(0)
        scale_3 = scale_3.detach().cpu().numpy().squeeze(0).squeeze(0)
        save_out = test_output.detach().cpu().numpy().squeeze(0).squeeze(0)
    return save_out, scale_1, scale_2, scale_3



def update_B(image, model):
    v_image = np.expand_dims(image, (0, 1))
    v_image = Variable(torch.Tensor(v_image)).cuda()
    with torch.no_grad():
        test_output = model(v_image)
        save_out = test_output.detach().cpu().numpy().squeeze(0).squeeze(0)
    return save_out