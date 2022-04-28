#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  28 03:12:09 2022

@author: kunal.swami

read_file, read_byte_data and shift_2d functions are modified from Sabour et al.'s Tensorflow code at https://github.com/Sarasra/models/tree/master/research/capsules
"""


import numpy as np
import pickle
import matplotlib.pyplot as plt

import sys
import os
import gzip

from torch.utils.data import Dataset

class MultiMNIST_Dataloader(Dataset):

    def __init__(self, is_train=True):
        self.data_dir = './Data/mnist/'

        self.MNIST_FILES = {
            'train': ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'),
            'test': ('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')
        }

        self.MNIST_RANGE = {
            'train': (0, 60000),
            'test': (0, 10000)
        }

        self.IMAGE_SIZE_PX = 28
        self.shift = 6
        self.pad = 4

        self.images = []
        self.targets = []

        if is_train:
            self.images, self.targets = self.read_byte_data(data_dir=self.data_dir, split='train')
        else:
            self.images, self.targets = self.read_byte_data(data_dir=self.data_dir, split='test')

        # pad to make all images in the dataset to make them 36x36
        self.padded_images = [np.pad(image, self.pad, 'constant')
                   for image in self.images]


    def __len__(self):
        return len(self.images) # size of MNIST dataset


    def __getitem__(self, item):
        i = np.random.randint(len(self.padded_images))
        j = np.random.randint(len(self.padded_images))
        while self.targets[i] == self.targets[j]:
            j = np.random.randint(len(self.padded_images))

        base_image, base_label = self.padded_images[i], self.targets[i]
        top_image, top_label = self.padded_images[j], self.targets[j]

        # random shifting of a digit in image before overlap
        random_shifts = np.random.randint(-self.shift, self.shift + 1, (2))

        base_shifted = self.shift_2d(base_image, random_shifts, self.shift).astype(np.uint8)
        top_shifted = self.shift_2d(top_image, random_shifts, self.shift).astype(np.uint8)
        merged = np.add(base_shifted, top_shifted, dtype=np.int32)
        merged = np.minimum(merged, 255).astype(np.uint8)

        return merged, base_shifted, top_shifted, base_label, top_label


    def read_file(self, file_bytes, header_byte_size, data_size):
        """Discards 4 * header_byte_size of file_bytes and returns data_size bytes."""
        file_bytes.read(4 * header_byte_size)
        return np.frombuffer(file_bytes.read(data_size), dtype=np.uint8)


    def read_byte_data(self, data_dir, split):
        """Extracts images and labels from MNIST binary file.
        Reads the binary image and label files for the given split. Generates a
        tuple of numpy array containing the pairs of label and image.
        The format of the binary files are defined at:
        http://yann.lecun.com/exdb/mnist/
        In summary: header size for image files is 4 * 4 bytes and for label file is
        2 * 4 bytes.
        Args:
        data_dir: String, the directory containing the dataset files.
        split: String, the dataset split to process. It can be one of train, test.
        Returns:
        A list of (image, label). Image is a 28x28 numpy array and label is an int.
        """
        image_file, label_file = (
            os.path.join(data_dir, file_name) for file_name in self.MNIST_FILES[split])
        start, end = self.MNIST_RANGE[split]
        with gzip.open(image_file, 'r') as f:
            images = self.read_file(f, 4, end * self.IMAGE_SIZE_PX * self.IMAGE_SIZE_PX)
            images = images.reshape(end, self.IMAGE_SIZE_PX, self.IMAGE_SIZE_PX)
        with gzip.open(label_file, 'r') as f:
            labels = self.read_file(f, 2, end)

        return images[start:], labels[start:]


    def shift_2d(self, image, shift, max_shift):
        """Shifts the image along each axis by introducing zero.
        Args:
        image: A 2D numpy array to be shifted.
        shift: A tuple indicating the shift along each axis.
        max_shift: The maximum possible shift.
        Returns:
        A 2D numpy array with the same shape of image.
        """
        max_shift += 1
        padded_image = np.pad(image, max_shift, 'constant')
        rolled_image = np.roll(padded_image, shift[0], axis=0)
        rolled_image = np.roll(rolled_image, shift[1], axis=1)
        shifted_image = rolled_image[max_shift:-max_shift, max_shift:-max_shift]
        return shifted_image

    def transform(self, merged, base_shifted, top_shifted, base_label, top_label):

        return merged, base_shifted, top_shifted, base_label, top_label


