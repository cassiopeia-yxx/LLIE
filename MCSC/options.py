# -*- coding: utf-8 -*-
# Auther : ML_XX
# Date : 2023/12/19 20:43
# File : Options.py
import argparse


def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="E:\Code\Python\denoising\R",
                        help='path to training input data')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--in_channel', type=int, help='the channel of input image', default=1)
    parser.add_argument('--patchSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--niter', type=int, default=200, help='total number of training epochs')
    # parser.add_argument('--iter', type=int, default=10, help='total number of proximal gradient descent iterations')
    parser.add_argument("--save_path", type=str, default="./results/enhancement", help='path to enhancement results')
    parser.add_argument('--DCU_T', type=int, default=10, help='the number of the DCU')
    parser.add_argument('--resume', type=int, default=0, help='continue to train from epoch')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
    parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
    parser.add_argument('--log_dir', default='./result_logs/', help='tensorboard logs')
    parser.add_argument('--model_dir', default='../MCSC/result_models', help='saving model')
    parser.add_argument('--manualSeed', default= 2, type=int, help='manual seed')
    return parser.parse_args()