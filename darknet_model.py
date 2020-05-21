import torch
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
import cv2
from utils import *

class dummyLayer(nn.Module):
    def __init__(self):
        super(dummyLayer, self).__init__()

class detector(nn.Module):
    def __init__(self, anchors):
        super(detector, self).__init__()
        self.anchors = anchors

def construct_cfg(cfgfile):
    """ 
    Building the network from configuration files     
    """
    config = open(cfgfile, 'r')
    file = config.read().split('\n')

    file = [line for line in file if len(line) > 0 and line[0] != '#']
    file = [line.lstrip().rstrip()for line in file]

    networkBlocks = []
    networkBlock = {}

    for x in file:
        if x[0] == '[':
            if len(networkBlock) != 0:
                networkBlocks.append(networkBlock)
                networkBlock = {}
            networkBlock['type'] = x[1:-1].rstrip()

        else:
            entity, value = x.split('=')
            networkBlock[entity.rstrip()] = value.lstrip()
        
    networkBlocks.append(networkBlock)

    return networkBlocks


nb = construct_cfg('/home/chandanv/Drive/Research/PytorchYolov3/cfg/yolov3.cfg')    
nb[0]
def build_network(networkBlocks):
    DNInfo = networkBlocks[0]
    modules = nn.ModuleList([])
    channels = 3
    filterTracker = []

    for i, x in enumerate(networkBlocks[1:]):
        seq_module = nn.Sequential()
        if (x['type'] == 'convolutional'):

            filters = int(x['filters'])
            kernelSize = int(x['size'])
            stride = int(x['stride'])
            pad = int(x['pad'])

            if pad:
                padding = (kernelSize - 1)/ 2
            else:
                padding = 0

            activation = x['activation']
            try:
                bn = int(x['batch_normalize'])
                bias = False
            except:
                bn = 0
                bias = True

            conv = nn.Conv2d(channels, filters, kernelSize, stride= stride, padding= padding, bias = bias)
            seq_module.add_module(f'conv_{i}', conv)

            if bn:
                bn = nn.BatchNorm2d(filters)
                seq_module.add_module(f'batch_norm_{i}', bn)

            if activation == 'leaky':
                activn = nn.LeakyReLU(filters)
                seq_module.add_module(f'leaky_{i}', activn)

        elif x['type'] == 'upsample':
            upsample = nn.Upsample(scale_factor= 2, mode= 'bilinear')
            seq_module.add_module(f'upsample_{i}', upsample)

        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')
            start = int(x['layers'][0])
            try:
                end = int(x['layers'][1])
            except:
                end = 0

            if start > 0:
                start = start - i
            if end > 0:
                end = end - i


            route = dummyLayer()
            seq_module.add_module(f'route_{i}', route)
