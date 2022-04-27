#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 20:36:13 2022

@author: saksham
"""

import os
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from torchvision import datasets, transforms
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torchvision.utils as tvutils

from prettytable import PrettyTable
import numpy as np

class Squash():
    '''
    Like a activation function but performed on whole layer output and not neuron/element wise
    '''
    
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
    def perform(self, s: torch.Tensor):
        
        s2 = (s ** 2).sum(dim=-1, keepdims=True)
        
        # Adding epsilon in case s2 become 0
        return (s2 / (1 + s2)) * (s / torch.sqrt(s2 + self.epsilon))
    
class Routing(nn.Module):
    
    def __init__(self, in_caps: int, out_caps: int, in_d: int, out_d: int, iterations: int):

        '''
        params
        in_caps : Input from previvous layer reshaped to form capsules, 32*8 channels divided in 2 parts
        32 as number of capsules (6*6*32) and 8 as input dimension

        out_caps : 10 for every class

        in_d : imput dimension (8)

        out_d : output dimension (length of each vector of output class)

        iterations : routing loop iterations
        '''

        super(Routing,self).__init__()

        #Intitialization with parameters required for routing
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.iterations = iterations
        self.softmax = nn.Softmax(dim=1)
        self.squash = Squash()

        # Weight matrix learned during training
        self.weight = nn.Parameter(torch.randn(in_caps, out_caps, in_d, out_d), requires_grad=True)


    def perform(self, u):

        self.weight = self.weight.to(u.device)

        # Prediction Vectors (Sumed over input dimentions)
        u_hat = torch.einsum('ijnm,bin->bijm', self.weight, u)

        b = u.new_zeros(u.shape[0], self.in_caps, self.out_caps)

        v = None

        for i in range(self.iterations):

            c = self.softmax(b)
            # Weighted sum over pridiction vectors
            s = torch.einsum('bij,bijm->bjm', c, u_hat)
            v = self.squash.perform(s)
            # agreement
            a = torch.einsum('bjm,bijm->bij', v, u_hat)

            b = b + a

        return v

class MarginLoss():
    
    def __init__(self, *, n_labels: int, lambda_: float = 0.5, m_positive: float = 0.9, m_negative: float = 0.1):

        '''
        params

        n_labels : Totals classes
        lambda_ : Use to decrease loss if wrong result so that training is not affected in start
        m_positive : weight of positive loss
        m_negative : weight of negative loss
        '''

        self.m_negative = m_negative
        self.m_positive = m_positive
        self.lambda_ = lambda_
        self.n_labels = n_labels

    def calculate(self, v: torch.Tensor, labels: torch.Tensor):

        v_norm = torch.sqrt((v ** 2).sum(dim=-1))

        labels = torch.eye(self.n_labels, device=labels.device)[labels]

        loss = labels * F.relu(self.m_positive - v_norm) + self.lambda_ * (1.0 - labels) * F.relu(v_norm - self.m_negative)

        return loss.sum(dim=-1).sum()


class DatasetHelper():
    def getDataSet(isTrain):

        return datasets.MNIST('./Data/mnist/', train=isTrain, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))


class DataParallel():

    def setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    def prepare(self, isTrain, rank, world_size, batch_size=128, pin_memory=False, num_workers=0):
        dataset = DatasetHelper.getDataSet(isTrain)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)

        return dataloader

    def cleanup(self):
        dist.destroy_process_group()

class Helper():

    def __init__(self):

        self.margin_loss = MarginLoss(n_labels=10)
        self.reconstruction_loss = nn.MSELoss(reduction='sum')


    def count_parameters(self, model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params


    def cost(self, caps: torch.Tensor, targets: torch.Tensor, reconstructions: torch.Tensor, data: torch.Tensor, isReconstruction = False) -> torch.Tensor:

        margin_loss = self.margin_loss.calculate(caps, targets)
        if isReconstruction == True:
            return  margin_loss + 0.0005 * self.reconstruction_loss(reconstructions, data)

        return margin_loss

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
        padded_image = np.pad(image, ((0, 0), (0, 0), (max_shift, max_shift), (max_shift, max_shift)), 'constant')
        rolled_image = np.roll(padded_image, shift[:][0], axis=2)
        rolled_image = np.roll(rolled_image, shift[:][1], axis=3)
        shifted_image = rolled_image[:, :, max_shift:-max_shift, max_shift:-max_shift]
        return shifted_image

    def transformData(self, data, batch_size):

        '''
        This function is used to transform MNIST data to 40x40 by padding and randomly shifting
        '''

        # undo normalization
        data = data * 0.3081 + 0.1307
        # transformations for shifted MNIST
        shift, max_shift = 6, 6
        # print(data.shape)
        data_numpy = data.cpu().detach().numpy()
        padded_data_numpy = np.pad(data_numpy, ((0, 0), (0, 0), (6, 6), (6, 6)), 'constant')
        # print(np.shape(padded_data_numpy))
        random_shifts = np.random.randint(-shift, shift + 1, (batch_size, 2))
        shifted_padded_data_numpy = self.shift_2d(padded_data_numpy, random_shifts, max_shift=6)
        # print(np.shape(shifted_padded_data_numpy))
        data = torch.from_numpy(shifted_padded_data_numpy)
        # redo normalization
        data = (data - 0.1307) / 0.3081
        #print('Input data shape: ', data.shape)

        return data


    def evaluate(self, network, epoch, batch_size, writer, rank=None, isShiftedMNIST=False):

        if rank:
            dev = rank
        elif torch.cuda.is_available():
            print("GPU available")
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")

        train_loader = torch.utils.data.DataLoader(DatasetHelper.getDataSet(True), batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(DatasetHelper.getDataSet(False), batch_size=batch_size)

        network.to(dev)
        network.eval()
        # Compute accuracy on training set
        count = 0
        train_running_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(train_loader):

                if isShiftedMNIST == True:
                    data = self.transformData(data, batch_size)

                data = data.to(dev)
                target = target.to(dev)

                caps, reconstructions, preds = network.forward(data)
                count += torch.sum(preds == target).detach().item()

                batch_loss = self.cost(caps, target, reconstructions, data, True)

                train_running_loss += batch_loss.item()

                # Logging reconstructed images
                grid = tvutils.make_grid(reconstructions)
                writer.add_image('train_recons_images', grid, (epoch+1))

        train_loss = train_running_loss / train_loader.dataset.data.size(0)
        train_accuracy = float(count) / train_loader.dataset.data.size(0)


        # Compute accuracy on test set
        count = 0
        test_running_loss = 0.0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):

                if isShiftedMNIST == True:
                    data = self.transformData(data, batch_size)

                data = data.to(dev)
                target = target.to(dev)

                caps, reconstructions, preds = network.forward(data)
                count += torch.sum(preds == target).detach().item()

                batch_loss = self.cost(caps, target, reconstructions, data, True)
                test_running_loss += batch_loss.item()

                grid = tvutils.make_grid(reconstructions)
                writer.add_image('test_recons_images', grid, (epoch+1))

        test_loss = test_running_loss / test_loader.dataset.data.size(0)
        test_accuracy = float(count) / test_loader.dataset.data.size(0)

        return train_accuracy, test_accuracy, train_loss, test_loss
        