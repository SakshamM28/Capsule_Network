#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 14:32:15 2022

@author: saksham
"""

import requests
import os
from zipfile import ZipFile

import scipy.io as sio
import numpy as np

import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from skimage.transform import resize

import torchvision.utils as tvutils


class affNISTData(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        
        self.dataset = images
        self.labels = labels
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        img_tensor = torch.from_numpy(self.dataset[idx])
        label_tensor = self.labels[idx]
        
        return img_tensor, label_tensor


def getDataset():
    
    # Tensorboard
    writer = SummaryWriter('runs/capsule_affnist_eval')

    # Test data URL

    test_URL = "https://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/test.mat.zip"
    data_dir = "./Data/affNIST/"
    
    file_name = "test.mat"
    
    #TODO Check data path or create
    
    if os.path.exists(os.path.join(data_dir + file_name)) == False:
        #Download Test Data
        response = requests.get(test_URL)
        zip_path = os.path.join(data_dir, "test.mat.zip")
        open(zip_path, "wb").write(response.content)
    
        with ZipFile(zip_path, 'r') as zip:
            zip.extractall(path=data_dir)
        
    data_path = os.path.join(data_dir, file_name)
    
    data = sio.loadmat(data_path)
    
    
    images = np.stack(data['affNISTdata']['image'].ravel()).transpose().reshape(-1,40,40,1).astype(np.float32)
    print("Images shape",images.shape)
    
    images_reized_l = []
    for i in range(0, len(images)):
        image_resized = resize(images[i], (28, 28 ),anti_aliasing=True)
        
        # Transposing make [1, 28,28] as feeded to network
        images_reized_l.append(image_resized.T)
        
        grid = tvutils.make_grid(torch.from_numpy(image_resized.reshape(28,28)))
        writer.add_image('resized_images', grid, i+1)
    print(len(images_reized_l))
    
    print(np.array(images_reized_l).shape)
    
    #  Labels as actual int 0-9
    labels = data['affNISTdata']['label_int']
    labels = np.stack(labels.ravel()).transpose().reshape(-1)
    labels.shape
    print("Labels shape",labels.shape)
    
    dataset = affNISTData(np.array(images_reized_l),labels)
    
    return dataset

if __name__ == '__main__':
    
    print(getDataset())