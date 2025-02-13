# -*- coding: utf-8 -*-
# Auther : ML_XX
# Date : 2024/1/11 13:13
# File : PWAS.py
import os
from uilts import *
import time
import numpy as np
from denoiser.network import IRCNN
from denoiser.option import options
from MCSC.model.network import DeNet
from MCSC.options import Options
from scipy.io import loadmat

"""
PWAS算法
"""
def PWAS(I, parameter):
    h, w = I.shape
    R = np.zeros((h, w))
    L = np.zeros((h, w))
    B = np.zeros((h, w))
    zero = np.zeros((h, w))
    temp_L = 0
    temp_R = 0
    temp_B = 0
    error_R = np.zeros((parameter.outer_iter, 1))
    error_B = np.zeros((parameter.outer_iter, 1))
    error_L = np.zeros((parameter.outer_iter, 1))
    errorL = 10
    errorR = 10
    errorB = 10
    I = np.asarray(I)
    mu = parameter.mu
    mcsc_opt = Options()
    denoiser_opt = options()
    os.environ["CUDA_VISIBLE_DEVICES"] = mcsc_opt.gpu_id
    mcsc_model = DeNet(mcsc_opt).cuda()
    mcsc_model.load_state_dict(torch.load(os.path.join(mcsc_opt.model_dir, 'best_model.pt').replace('\\', '/')))
    mcsc_model.eval()

    ircnn_model = IRCNN().cuda()
    ircnn_model.load_state_dict(torch.load(os.path.join(denoiser_opt.model_dir, 'best_model.pt').replace('\\', '/')))
    ircnn_model.eval()

    start_time = time.time()
    for i in range(parameter.outer_iter):
        # update L
        L_start_time = time.time()
        L = update_s((1 / (mu + 1))*(I + mu * B), R, parameter.lambda3)
        error_L[i] = np.linalg.norm((L - temp_L)) / np.linalg.norm(L)
        errorL = error_L[i]
        temp_L = L
        L_end_time = time.time()

        # update R
        R_start_time = time.time()
        R, scale_1, scale_2, scale_3 = update_z((1 / (mu + 1))*(I + mu * B) - L, mcsc_model, mcsc_opt)
        error_R[i] = np.linalg.norm((R - temp_R)) / np.linalg.norm(R)
        errorR = error_R[i]
        temp_R = R
        R_end_time = time.time()

        # update B
        B_start_time = time.time()
        B = update_B(L + R, ircnn_model)
        error_B[i] = np.linalg.norm((B - temp_B)) / np.linalg.norm(B)
        errorB = error_B[i]
        temp_B = B
        B_end_time = time.time()

        # if abs(error_L[i]) < parameter.tol and abs(error_R[i]) < parameter.tol:
        #     break
        # print("第{}次迭代，R的耗时:{}, B的耗时:{}, L的耗时:{}".format(i, R_end_time - R_start_time, B_end_time - B_start_time, L_end_time - L_start_time))
        errorL_float = float(errorL)
        errorR_float = float(errorR)
        errorB_float = float(errorB)

        # print('iter = {0}, error_L = {1:.4f}, error_R = {2:.4f}, error_B = {3:.4f}'.format(i, errorL_float, errorR_float, errorB_float))
        # print('iter = {0}, error_L = {1:.4f}, error_R = {2:.4f}, error_B = {3:.4f}'.format(i, errorL, errorR, errorB))
    end_time = time.time()
    print("PWAS算法结束，耗时:{}".format(end_time - start_time))
    # R = np.asnumpy(R)
    # L = np.asnumpy(L)
    # error_L = np.asnumpy(error_L)
    # error_R = np.asnumpy(error_R)
    return R, L, error_L, error_R, error_B
