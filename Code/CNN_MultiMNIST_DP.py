# -*- coding: utf-8 -*-

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

from Modules_CNN import Helper, DataParallel, DatasetHelper
from MultiMNIST_Dataloader import MultiMNIST_Dataloader

class MultiMnistCNN(nn.Module):

    def __init__(self):
        super(MultiMnistCNN, self).__init__()

        # Define the architecture
        self.model = nn.Sequential()

        self.model.add_module('conv1', nn.Conv2d(in_channels=1, out_channels=512, kernel_size=9, stride=1))  # (28, 28, 512)
        self.model.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))  # (14, 14, 512)
        self.model.add_module('activation1', nn.ReLU())

        self.model.add_module('conv2', nn.Conv2d(in_channels=512, out_channels=256, kernel_size=5, stride=1))  # (10, 10, 256)
        self.model.add_module('pool2', nn.MaxPool2d(kernel_size=2, stride=2))  # (5, 5, 256)
        self.model.add_module('activation2', nn.ReLU())

        self.model.add_module('flatten', nn.Flatten())

        self.model.add_module('linear1', nn.Linear(in_features=6400, out_features=1024))
        self.model.add_module('activation3', nn.ReLU())

        self.model.add_module('linear2', nn.Linear(in_features=1024, out_features=10))
        self.model.add_module('activation4', nn.LogSoftmax(dim=1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        :param images: (batch_size, height, width) Images to be classified
        :return predictions: (batch_size, 10) Output predictions (log probabilities)
        """
        return self.model(images)


def main(rank, world_size, batch_size, num_epochs, learning_rate, model_path, num_exp):
    
    print(rank, world_size, batch_size, num_epochs, learning_rate, model_path, num_exp)
    
    helper = Helper()

    if rank == 0:
        # Tensorboard
        writer = SummaryWriter('runs/experiment_' + num_exp)

    # Data Parallelism for Multiple GPU
    dataParallel = DataParallel()
    # setup the process groups
    dataParallel.setup(rank, world_size)
    # Set up the data loader
    train_loader = dataParallel.prepare(True, rank, world_size, batch_size, num_workers=4, is_MultiMNIST=True)

    ## Load Full Test data for evaluation
    test_loader = torch.utils.data.DataLoader(MultiMNIST_Dataloader(is_train=False), batch_size=batch_size)

    if rank == 0:
        print('Training dataset size: ', len(train_loader.dataset))
        print('Test dataset size: ', len(test_loader.dataset))

    # Set up the network and optimizer
    network = MultiMnistCNN()
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
        for batch_idx, (merged, base_shifted, top_shifted, base_label, top_label) in enumerate(train_loader):
            merged = merged.to(rank)
            #base_shifted = base_shifted.to(rank)
            #top_shifted = top_shifted.to(rank)
            base_label = base_label.to(rank)
            top_label = top_label.to(rank)

            # Get the predictions
            preds = network.forward(merged)
            #print(preds.shape)

            # Compute the loss
            loss_1 = helper.cost(preds, base_label)
            loss_2 = helper.cost(preds, top_label)
            loss = 0.5 * (loss_1 + loss_2)

            # Take a gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # LR decay after every 2 epochs, original paper applies after ~3 epochs
        if (epoch+1)%2 ==0:
            lr_scheduler.step()
        
        # Calculate accuracies on whole dataset
        if rank == 0:
            train_accuracy, test_accuracy, train_loss, test_loss= helper.evaluate(network, epoch, batch_size, writer, rank, isMultiMNIST=True)
        #else:
            #train_accuracy, test_accuracy, train_loss, test_loss = helper.evaluate(network, epoch, batch_size, None, rank, isMultiMNIST=True)

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

        if rank == 0:
            train_acc_l.append(train_accuracy)
            test_acc_l.append(test_accuracy)

        if rank == 0:
            if test_accuracy == max(test_acc_l):
                best_epoch = epoch + 1

                # Saving the model with best test accuracy till current epoch
                torch.save(network.state_dict(), model_path + 'cnn_multimnist_' + str(num_epochs) + '_' + str(epoch+1) + '.pt')

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
    num_exp = sys.argv[3]

    model_path = 'saved_model/' + num_exp + '/'
    Path(model_path).mkdir(parents=True, exist_ok=True)

    # Put no. of GPU's used
    world_size = 2
    mp.spawn(
        main,
        args=(world_size, batch_size, num_epochs, learning_rate, model_path, num_exp),
        nprocs=world_size
    )
    