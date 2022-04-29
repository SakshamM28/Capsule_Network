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

from torchvision import transforms

class affNISTData(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform):
        
        self.dataset = images
        self.labels = labels
        self.transform=transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        img = self.dataset[idx]
        label = self.labels[idx]
        
        img_norm = self.transform(img)

        return img_norm, label


def getDataset(isResized=False):

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
    #print(data)

    images = np.stack(data['affNISTdata']['image'].ravel()).transpose().reshape(-1,40,40,1)
    print("Images shape", images.shape)

    images_reized_l = []
    for i in range(0, len(images)):
        if isResized == True:
            #print(" Resizing images to 28x28 for mnist")
            image_resized = resize(images[i], (28, 28 ),anti_aliasing=True)
        else:
            image_resized = images[i]

        images_reized_l.append(image_resized)

    print(len(images_reized_l))
    
    print(np.array(images_reized_l).shape)
    
    #  Labels as actual int 0-9
    labels = data['affNISTdata']['label_int']
    labels = np.stack(labels.ravel()).transpose().reshape(-1)
    print("Labels shape",labels.shape)
    
    resized_images = np.array(images_reized_l)

    # TODO: Check if Resize function can be used, nedd PIL image
    transform=transforms.Compose([
            #transforms.Resize((28,28)),
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,)) # enable for CNN (by mistake trained CNN with this!)
            ])

    dataset = affNISTData(resized_images, labels, transform)
    
    return dataset

if __name__ == '__main__':
    
    # Tensorboard
    writer = SummaryWriter('runs/capsule_affnist_eval')

    data = getDataset(False)

    for i in range(1,50):
        image = data.__getitem__(i)[0]

        print(torch.max(image), image.size())

        #image = image * 0.3081 + 0.1307
        grid = tvutils.make_grid(image)
        writer.add_image('resized_images', grid, i)

    writer.flush()
    writer.close()