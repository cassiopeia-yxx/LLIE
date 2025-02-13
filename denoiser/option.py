# -*- coding: utf-8 -*-
# Auther : ML_XX
# Date : 2024/1/9 10:18
# File : option.py
import argparse


def options():
    parser = argparse.ArgumentParser(description='dual_denoise')
    parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
    parser.add_argument('--save_path', type=str, default='./interactive', help='dataset path')
    parser.add_argument('--model_path', type=str, default='./model', help='save model path')
    parser.add_argument('--in_channel', type=int, default=1, help='the channel of input image')
    parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
    parser.add_argument("--milestone", type=int, default=[25, 50, 75], help="When to decay learning rate")
    parser.add_argument('--iter', type=int, default=10, help='total number of training epochs')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--patchSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--nEpoch', type=int, default=100, help='total number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--resume', type=int, default=0, help='continue to train from epoch')
    parser.add_argument('--log_dir', default='./result_logs/', help='tensorboard logs')
    parser.add_argument('--model_dir', default='../denoiser/result_models/', help='saving model')
    return parser.parse_args()