import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import PIL.Image as Image
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable
import os
import sys
import torch.nn as nn
import cv2
# from skimage.color import hed2rgb, rgb2hed
import random
import glob
import argparse
from PIL import Image


class data_loader(Dataset):
    """
    Dataset to read image and label for training
    """
    def __init__(self, imgs, transform=None, class_name=None):
        self.imgs = imgs
        self.transform = transform
        self.class_name = class_name

    def convert_lb(self, lb):
        if lb in {4, 5, 6}:
            lb -= 1
        return lb

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        img = self.transform(img)

        lb = int(self.imgs[index][-5])
        lb = self.convert_lb(lb)
        return img, lb   # *_{lb}.png --> extract the label

    def __len__(self):
        return len(self.imgs)

def parallelize_model(model):
    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        cudnn.benchmark = True
    return model


def unparallelize_model(model):
    try:
        while 1:
            # to avoid nested dataparallel problem
            model = model.module
    except AttributeError:
        pass
    return model


def cvt_to_gpu(X):
    return Variable(X.cuda()) if torch.cuda.is_available() \
    else Variable(X)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    cnt = 0
    for inputs, targets in dataloader:
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)
            sys.stdout.flush()

        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    print('mean, std: ', mean, std)
    return mean, std

def convert_lb(lb):
    lb = int(lb)
    if lb == 2 or lb == 3:
        return lb - 1
    return lb

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images: # item is the path to .png file
        count[convert_lb(item[-5])] += 1

    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[convert_lb(val[-5])]
    return weight

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
