from __future__ import print_function 
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import WIDERFace_ROOT , WIDERFace_CLASSES as labelmap
from PIL import Image
from data import WIDERFaceDetection, WIDERFaceAnnotationTransform, WIDERFace_CLASSES, WIDERFace_ROOT, BaseTransform , TestBaseTransform
from data import *
import torch.utils.data as data
from face_ssd import build_ssd
#from resnet50_ssd import build_sfd
import pdb
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import time

from utils.infer_utils import *


plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='DSFD:Dual Shot Face Detector')
parser.add_argument('--trained_model', default='weights/WIDERFace_DSFD_RES152.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval_tools/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.1, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--img_root', default='./data/worlds-largest-selfie.jpg', help='Location of test images directory')
parser.add_argument('--widerface_root', default=WIDERFace_ROOT, help='Location of WIDERFACE root directory')

args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)





if __name__ == '__main__':
    test_oneimage(args)
