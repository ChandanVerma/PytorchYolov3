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

    config = open(cfgfile, 'r')
    file = config.read().split('\n')

    file = [line for line in file if len(line) > 0 and line[0] != '#']
    file = [line.lstrip().rstrip()for line in file]

    networkBlocks = []
    networkBlock = []

    