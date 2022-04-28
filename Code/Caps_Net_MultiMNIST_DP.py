#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  10 16:00:08 2022

@author: saksham
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.tensorboard import SummaryWriter

from Modules import Squash, Routing, Helper, DataParallel, DatasetHelper

class ShiftedMNISTCapsuleNetworkModel(nn.Module):
    #TODO take dynamic parameters for routing, input size etc
    def __init__(self):
        super(ShiftedMNISTCapsuleNetworkModel, self).__init__()
        
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=32 * 8, kernel_size=9, stride=2, padding=0)
        
        self.squash = Squash()

        self.digit_capsules = Routing(32 * 12 * 12, 10, 8, 16, 3)
        
        
        self.decoder = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1296),
            nn.Sigmoid()
        )
        
    def forward(self, data: torch.Tensor):
        
        x = F.relu(self.conv1(data))
        x = self.conv2(x)
        
        caps = x.view(x.shape[0], 8, 32 * 10 * 10).permute(0, 2, 1)
        caps = self.squash.perform(caps)
        caps = self.digit_capsules.perform(caps)
        
        
        with torch.no_grad():
            pred = (caps ** 2).sum(-1).argmax(-1)
            mask = torch.eye(10, device=data.device)[pred]
            
        reconstructions = self.decoder((caps * mask[:, :, None]).view(x.shape[0], -1))
        
        reconstructions = reconstructions.view(-1, 1, 40, 40)
        
        return caps, reconstructions, pred
    
def main(rank, world_size, batch_size, num_epochs, learning_rate, model_path, num_exp):
    
    print(rank, world_size, batch_size, num_epochs, learning_rate, model_path, num_exp)
    
    helper = Helper()

    if rank == 0:
        # Tensorboard
        writer = SummaryWriter('runs/capsule_mnist_experiment_' + str(num_epochs) + "_" + str(num_exp))

    # Data Parallelism for Multiple GPU
    dataParallel = DataParallel()
    # setup the process groups
    dataParallel.setup(rank, world_size)
    # Set up the data loader
    train_loader = dataParallel.prepare(True, rank, world_size, batch_size)

    ## Load Full Test data for evaluation
    test_loader = torch.utils.data.DataLoader(DatasetHelper.getDataSet(False), batch_size=batch_size)

    if rank == 0:
        print('Training dataset size: ', train_loader.dataset.data.size(0))
        print('Test dataset size: ', test_loader.dataset.data.size(0))

    # Set up the network and optimizer
    network = ShiftedMNISTCapsuleNetworkModel()
    network.to(rank)
    network= DDP(network, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    if rank == 0:
        table, total_params = helper.count_parameters(network)
        print(table)
        print('Total trainable parameters: ', total_params)

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    
    train_acc_l = []
    test_acc_l = []
    best_epoch = 0
    # Train the network
    for epoch in range(num_epochs):
        
        train_loader.sampler.set_epoch(epoch)

        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(rank)
            target = target.to(rank)

            # Get the predictions
            caps, reconstructions, preds = network.forward(data)
            #print(preds.shape)

            # Compute the loss
            loss = helper.cost(caps, target, reconstructions, data, True)

            # Take a gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # LR decay after every 2 epochs, original paper applies after ~3 epochs
        if (epoch+1)%2 ==0:
            lr_scheduler.step()
        
        # Calculate accuracies on whole dataset
        if rank == 0:
            train_accuracy, test_accuracy, train_loss, test_loss= helper.evaluate(network, epoch, batch_size, writer, rank, isShiftedMNIST=False)
        else:
            train_accuracy, test_accuracy, train_loss, test_loss = helper.evaluate(network, epoch, batch_size, None, rank, isShiftedMNIST=False)

        if rank == 0:
            print('Epoch:', '{:3d}'.format(epoch + 1),
                  '\tTraining Accuracy:', '{:10.5f}'.format(train_accuracy),
                  '\tTraining Loss:', '{:10.5f}'.format(train_loss),
                  '\tTesting Accuracy:', '{:10.5f}'.format(test_accuracy),
                  '\tTesting Loss:', '{:10.5f}'.format(test_loss))

        if rank == 0:
            # Log LR
            writer.add_scalar(str(rank)+': learning rate', lr_scheduler.get_last_lr()[0], (epoch + 1))
            # log the training loss
            writer.add_scalar(str(rank)+': training epoch loss', train_loss, (epoch+1))
            # log the evaluation loss
            writer.add_scalar(str(rank)+': evaluation epoch loss', test_loss, (epoch+1))
            # log accuracies
            writer.add_scalar(str(rank)+': Training epoch Accuracy', train_accuracy, (epoch+1))
            writer.add_scalar(str(rank)+': Testing epoch Accuracy', test_accuracy, (epoch+1))
        
        train_acc_l.append(train_accuracy)
        test_acc_l.append(test_accuracy)
        
        if test_accuracy == max(test_acc_l):
            best_epoch = epoch + 1

            # Saving the model with best test accuracy till current epoch
            torch.save(network.state_dict(), model_path + 'caps_net_multi_mnist_' + str(num_epochs) + '_' + str(epoch+1) + '.pt')

    if rank == 0:
        writer.flush()
        writer.close()


    # Display Max Accuracy
    if rank == 0:
        print('Training completed!')
        print('Max Train Accuracy : ', max(train_acc_l))
        print('Max Test Accuracy : ', max(test_acc_l))
        print('Best Test Accuracy epoch: ', best_epoch)
    
    
    dataParallel.cleanup()


import torch.multiprocessing as mp
import os
from pathlib import Path

if __name__ == '__main__':
    
    # Control variables
    batch_size = int(sys.argv[1])
    num_epochs = int(sys.argv[2])
    learning_rate = 1e-3
    num_exp = int(sys.argv[3])

    model_path = 'saved_model/caps_multi_mnist/'
    Path(model_path).mkdir(parents=True, exist_ok=True)
    
    # Put no. of GPU's used
    world_size = 2
    mp.spawn(
        main,
        args=(world_size, batch_size,num_epochs,learning_rate, model_path, num_exp),
        nprocs=world_size
    )
    