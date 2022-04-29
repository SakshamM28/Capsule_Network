# -*- coding: utf-8 -*-


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

from MultiMNIST_Dataloader import MultiMNIST_Dataloader

class DataParallel():

    def setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    def prepare(self, isTrain, rank, world_size, batch_size=128, pin_memory=False, num_workers=0, is_MultiMNIST=False):
        if is_MultiMNIST:
            dataset = MultiMNIST_Dataloader(is_train=isTrain)
        else:
            dataset = DatasetHelper.getDataSet(isTrain)

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)

        return dataloader

    def cleanup(self):
        dist.destroy_process_group()

class DatasetHelper():
    def getDataSet(isTrain):

        return datasets.MNIST('./Data/mnist/', train=isTrain, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))


class Helper():

    def __init__(self):
        self.loss = nn.NLLLoss(reduction='sum')
        
        
    def cost(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param predictions: (batch_size, 10) Output predictions (log probabilities)
        :param targets: (batch_size, 1) Target classes
        :return cost: The negative log-likelihood loss
        """
        return self.loss(predictions, targets.view(-1))


    def count_parameters(self, model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        #print(table)
        #print(f"Total Trainable Params: {total_params}")
        return table, total_params

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

    def transformData_ShiftedMNIST(self, data, batch_size):

        '''
        This function is used to transform MNIST data to 40x40 by padding and randomly shifting
        '''

        # undo normalization
        #data = data * 0.3081 + 0.1307
        # transformations for shifted MNIST
        shift, max_shift = 6, 6
        # print(data.shape)
        data_numpy = data.cpu().detach().numpy()
        padded_data_numpy = np.pad(data_numpy, ((0, 0), (0, 0), (6, 6), (6, 6)), 'constant')
        # print(np.shape(padded_data_numpy))
        random_shifts = np.random.randint(-shift, shift + 1, (batch_size, 2))
        shifted_padded_data_numpy = self.shift_2d(padded_data_numpy, random_shifts, max_shift=max_shift)
        # print('Data shape after transformation: ', np.shape(shifted_padded_data_numpy))
        data = torch.from_numpy(shifted_padded_data_numpy)
        # redo normalization
        #data = (data - 0.1307) / 0.3081

        return data

    def transformData_MNIST(self, data, batch_size):

        '''
        Simialr to original CapsNet paper, this function is used to apply shift upto 2 pixels in each direction with zero padding
        '''

        # undo normalization
        #data = data * 0.3081 + 0.1307
        # transformations for shifted MNIST
        shift, max_shift = 2, 2
        # print(data.shape)
        data_numpy = data.cpu().detach().numpy()
        random_shifts = np.random.randint(-shift, shift + 1, (batch_size, 2))
        shifted_data_numpy = self.shift_2d(data_numpy, random_shifts, max_shift=max_shift)
        #print('Data shape after transformation: ', np.shape(shifted_data_numpy))
        data = torch.from_numpy(shifted_data_numpy)
        # redo normalization
        #data = (data - 0.1307) / 0.3081

        return data


    def evaluate(self, network, epoch, batch_size, writer=None, rank=None, isShiftedMNIST=False, isMultiMNIST=False):

        if rank:
            dev = rank
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
            if rank == 0:
                print("GPU available")
        else:
            dev = torch.device("cpu")
        if isMultiMNIST:
            train_loader = torch.utils.data.DataLoader(MultiMNIST_Dataloader(is_train=False), batch_size=batch_size)
            test_loader = torch.utils.data.DataLoader(MultiMNIST_Dataloader(is_train=False), batch_size=batch_size)
        else:
            train_loader = torch.utils.data.DataLoader(DatasetHelper.getDataSet(True), batch_size=batch_size)
            test_loader = torch.utils.data.DataLoader(DatasetHelper.getDataSet(False), batch_size=batch_size)

        network.to(dev)
        network.eval()
        # Compute accuracy on training set
        count = 0
        train_running_loss = 0.0

        if isMultiMNIST:
            with torch.no_grad():
                for batch_idx, (merged, base_shifted, top_shifted, base_label, top_label) in enumerate(train_loader):
                    merged = merged.to(rank)
                    #base_shifted = base_shifted.to(rank)
                    #top_shifted = top_shifted.to(rank)
                    base_label = base_label.to(rank)
                    top_label = top_label.to(rank)

                    preds = network.forward(merged)
                    pred = torch.topk(preds, k=2)[1]
                    pred_1 = torch.squeeze(pred.narrow(1, 0, 1))
                    pred_2 = torch.squeeze(pred.narrow(1, 1, 1))
                    print('New pred: ', pred, pred.shape, pred_1, pred_1.shape, pred_2, pred_2.shape)
                    count += torch.sum(torch.logical_and(pred_1 == base_label, pred_2 == top_label)).detach().item()

                    batch_loss_1 = self.cost(preds, base_label)
                    batch_loss_2 = self.cost(preds, top_label)
                    batch_loss = 0.5 * (batch_loss_1 + batch_loss_2)

                    train_running_loss += batch_loss.item()

            train_loss = train_running_loss / len(train_loader.dataset)
            train_accuracy = float(count) / len(train_loader.dataset)
        else:
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(train_loader):

                    if isShiftedMNIST == True:
                        data = self.transformData_ShiftedMNIST(data, batch_size)

                    data = data.to(dev)
                    target = target.to(dev)

                    preds = network.forward(data)
                    _, pred = torch.max(preds, dim=1)
                    count += torch.sum(pred == target).detach().item()

                    batch_loss = self.cost(preds, target)

                    train_running_loss += batch_loss.item()

            train_loss = train_running_loss / train_loader.dataset.data.size(0)
            train_accuracy = float(count) / train_loader.dataset.data.size(0)


        # Compute accuracy on test set
        count = 0
        test_running_loss = 0.0

        if isMultiMNIST:
            with torch.no_grad():
                for batch_idx, (merged, base_shifted, top_shifted, base_label, top_label) in enumerate(test_loader):
                    merged = merged.to(rank)
                    base_shifted = base_shifted.to(rank)
                    top_shifted = top_shifted.to(rank)
                    base_label = base_label.to(rank)
                    top_label = top_label.to(rank)

                    preds = network.forward(merged)
                    pred = torch.topk(preds, dim=1)
                    pred_1 = torch.squeeze(pred.narrow(1, 0, 1))
                    pred_2 = torch.squeeze(pred.narrow(1, 1, 1))
                    print('New pred: ', pred, pred.shape, pred_1, pred_1.shape, pred_2, pred_2.shape)
                    count += torch.sum(torch.logical_and(pred_1 == base_label, pred_2 == top_label)).detach().item()

                    batch_loss_1 = self.cost(preds, base_label)
                    batch_loss_2 = self.cost(preds, top_label)
                    batch_loss = 0.5 * (batch_loss_1 + batch_loss_2)

                    test_running_loss += batch_loss.item()

            test_loss = test_running_loss / test_loader.dataset.data.size(0)
            test_accuracy = float(count) / test_loader.dataset.data.size(0)
        else:
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(test_loader):

                    if isShiftedMNIST == True:
                        data = self.transformData_ShiftedMNIST(data, batch_size)

                    data = data.to(dev)
                    target = target.to(dev)

                    preds = network.forward(data)
                    _, pred = torch.max(preds, dim=1)
                    count += torch.sum(pred == target).detach().item()

                    batch_loss = self.cost(preds, target)

                    test_running_loss += batch_loss.item()

            test_loss = test_running_loss / test_loader.dataset.data.size(0)
            test_accuracy = float(count) / test_loader.dataset.data.size(0)

        return train_accuracy, test_accuracy, train_loss, test_loss
