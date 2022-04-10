#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 18:15:35 2022

@author: kunal18

Modified from Sabour et al.'s Tensorflow code at https://github.com/Sarasra/models/tree/master/research/capsules
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

import sys
import os
import gzip


MNIST_FILES = {
    'train': ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'),
    'test': ('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')
}

MNIST_RANGE = {
    'train': (0, 60000),
    'test': (0, 10000)
}

IMAGE_SIZE_PX = 28


def read_file(file_bytes, header_byte_size, data_size):
    """Discards 4 * header_byte_size of file_bytes and returns data_size bytes."""
    file_bytes.read(4 * header_byte_size)
    return np.frombuffer(file_bytes.read(data_size), dtype=np.uint8)


def read_byte_data(data_dir, split):
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
      os.path.join(data_dir, file_name) for file_name in MNIST_FILES[split])
    start, end = MNIST_RANGE[split]
    with gzip.open(image_file, 'r') as f:
        images = read_file(f, 4, end * IMAGE_SIZE_PX * IMAGE_SIZE_PX)
        images = images.reshape(end, IMAGE_SIZE_PX, IMAGE_SIZE_PX)
    with gzip.open(label_file, 'r') as f:
        labels = read_file(f, 2, end)

    return list(zip(images[start:], labels[start:]))


def shift_2d(image, shift, max_shift):
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


def shift_write_multi_mnist(input_dataset, shift, pad, num_pairs, split='train'):
    """Pads the data by adding zeros. Shifts all images randomly. For each image
    randomly selects a set of other images with different label as its pair.
    Aggregates the image pair with a maximum pixel value of 255.

    Args:
    input_dataset: A list of tuples containing corresponding images and labels.
    shift: Integer, the shift range for images.
    pad: Integer, the number of pixels to be padded.
    num_pairs: Integer, number of pairs of images generated for each input
      image.
    """
    multi_mnist_dataset = []

    num_images = len(input_dataset)
    print('Len of input_dataset:', num_images)
    # random shifting of a digit in image before overlap
    random_shifts = np.random.randint(-shift, shift + 1,
                                      (num_images, num_pairs + 1, 2))
    # pad to make all images in the dataset 36x36
    dataset = [(np.pad(image, pad, 'constant'), label)
             for (image, label) in input_dataset]

    for i, (base_image, base_label) in enumerate(dataset):
        base_shifted = shift_2d(base_image, random_shifts[i, 0, :], shift).astype(np.uint8)
        choices = np.random.choice(num_images, 2 * num_pairs, replace=False)
        chosen_dataset = []
        for choice in choices:
            if dataset[choice][1] != base_label:
                chosen_dataset.append(dataset[choice])
        for j, (top_image, top_label) in enumerate(chosen_dataset[:num_pairs]):
            top_shifted = shift_2d(top_image, random_shifts[i, j + 1, :], shift).astype(np.uint8)
            merged = np.add(base_shifted, top_shifted, dtype=np.int32)
            merged = np.minimum(merged, 255).astype(np.uint8)
            """ Visualization to verify
            plt.imshow(merged)
            plt.show()
            """
            data_instance = {
                'height': IMAGE_SIZE_PX + 2 * pad,
                'width': IMAGE_SIZE_PX + 2 * pad,
                'depth': 1,
                'label_1': base_label,
                'label_2': top_label,
                'image_raw_1': base_shifted,
                'image_raw_2': top_shifted,
                'merged_raw': merged,
            }
            multi_mnist_dataset.append(data_instance)

        if i!=0 and i%1000==0:
            print('Finished ' + str(i) + ' of 60000')

    with open('./multi_mnist_'+split+'_data.pickle', 'wb') as output:
        pickle.dump(multi_mnist_dataset, output)


def main(num_pairs=100):
    data = read_byte_data(data_dir='/Users/kunal/Downloads/capsule_networks_exps/MNIST', split='train')
    shift_write_multi_mnist(data, shift=6, pad=4, num_pairs=num_pairs, split='train')
    data = read_byte_data(data_dir='/Users/kunal/Downloads/capsule_networks_exps/MNIST', split='test')
    shift_write_multi_mnist(data, shift=6, pad=4, num_pairs=num_pairs, split='test')


if __name__ == '__main__':
    # Control variables
    num_pairs = int(sys.argv[1])
    main(num_pairs)