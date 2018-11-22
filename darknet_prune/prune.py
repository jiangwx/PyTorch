import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms, datasets
import torchvision
from torch.utils.data import DataLoader,Dataset
from PIL import Image

import math
import time
import numpy as np

def get_layer_fm_index(index, conv_cfg):
    layer_index = 0
    fm_index = 0
    while (index >= conv_cfg[layer_index]):
        index = index - conv_cfg[layer_index]
        layer_index = layer_index + 1
    fm_index = index
    return layer_index, fm_index

def get_index(layer_index, fm_index, conv_cfg):
    index = 0
    for i in range(layer_index):
        index = index + conv_cfg[i]
    index = index + fm_index
    return index

def generate_pruned_cfg(loss_rank, cfg, prune_cnt):
    conv_cfg = []  # store the fm cnt of each conv layer
    pruned_layer_cfg = [[]]  # store the remain fm index of each conv layer
    pruned_model_cfg = cfg  # store pruned cfg to generate new model

    for l in cfg:
        if l != 'M':
            conv_cfg.append(l[0])

    for i in range(len(cfg) - cfg.count('M') - 1):
        pruned_layer_cfg.append([])

    fm_cnt = len(loss_rank) - prune_cnt  # remained fm count

    for i in range(fm_cnt):
        layer_index, fm_index = get_layer_fm_index(loss_rank[i], conv_cfg)
        pruned_layer_cfg[layer_index].append(fm_index)

    layer_index = 0
    for i in range(len(cfg)):
        if cfg[i][0] != 'M':
            pruned_model_cfg[i][0] = len(pruned_layer_cfg[layer_index])
            layer_index += 1
    return pruned_model_cfg, pruned_layer_cfg

def pruned_model_init(model, pruned_model, pruned_layer_cfg):

    conv_layer_index = 0

    for layer_index, pruned_layer in pruned_model.feature._modules.items():

        _, layer = model.feature._modules.items()[layer_index]

        if isinstance(pruned_layer, nn.Conv2d):

            if (conv_layer_index == 0):
                weight = layer.weight.data.cpu().numpy()
                pruned_weight = layer.weight.data.cpu().numpy()[:len(pruned_layer_cfg[conv_layer_index])]
                print pruned_weight.shape

                out_channel = 0
                for i in pruned_layer_cfg[conv_layer_index]:
                    pruned_weight[out_channel] = weight[i]
                    out_channel = out_channel + 1

                pruned_layer.weight.data = torch.from_numpy(pruned_weight).cuda()

            elif (conv_layer_index == 17):
                weight = layer.weight.data.cpu().numpy()
                pruned_weight = layer.weight.data.cpu().numpy()[:, :len(pruned_layer_cfg[conv_layer_index - 1])]
                print pruned_weight.shape

                in_channel = 0
                for i in pruned_layer_cfg[conv_layer_index - 1]:
                    pruned_weight[:, in_channel] = weight[:, i]
                    in_channel = in_channel + 1

                pruned_layer.weight.data = torch.from_numpy(pruned_weight).cuda()

            else:
                weight = layer.weight.data.cpu().numpy()
                pruned_weight = layer.weight.data.cpu().numpy()[:len(pruned_layer_cfg[conv_layer_index]), :len(pruned_layer_cfg[conv_layer_index - 1])]
                print pruned_weight.shape

                in_channel = 0
                out_channel = 0
                for i in pruned_layer_cfg[conv_layer_index]:
                    for j in pruned_layer_cfg[conv_layer_index - 1]:
                        pruned_weight[out_channel, in_channel] = weight[i, j]
                        in_channel = in_channel + 1
                    in_channel = 0
                    out_channel = out_channel + 1

                pruned_layer.weight.data = torch.from_numpy(pruned_weight).cuda()
            continue

        elif isinstance(pruned_layer, nn.BatchNorm2d):

            if (conv_layer_index < 18):

                pruned_weight = np.zeros(len(pruned_layer_cfg[conv_layer_index]))
                pruned_bias = np.zeros(len(pruned_layer_cfg[conv_layer_index]))
                pruned_mean = np.zeros(len(pruned_layer_cfg[conv_layer_index]))
                pruned_var = np.zeros(len(pruned_layer_cfg[conv_layer_index]))

                channel = 0
                for i in pruned_layer_cfg[conv_layer_index]:
                    pruned_weight[channel] = layer.weight.data.cpu().numpy()[i]
                    pruned_bias[channel] = layer.bias.data.cpu().numpy()[i]
                    pruned_mean[channel] = layer.running_mean.data.cpu().numpy()[i]
                    pruned_var[channel] = layer.running_var.data.cpu().numpy()[i]
                    channel = channel + 1
                pruned_layer.weight.data = torch.from_numpy(pruned_weight).cuda()
                pruned_layer.bias.data = torch.from_numpy(pruned_bias).cuda()
                pruned_layer.running_mean.data = torch.from_numpy(pruned_mean).cuda()
                pruned_layer.running_var.data = torch.from_numpy(pruned_var).cuda()

            conv_layer_index += 1