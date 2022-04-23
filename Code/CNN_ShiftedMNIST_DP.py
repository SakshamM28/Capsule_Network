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

import numpy as np

class ShiftedMnistCNN(nn.Module):

    def __init__(self):
        super(ShiftedMnistCNN, self).__init__()

        # Define the architecture
        self.model = nn.Sequential()

        self.model.add_module('conv1', nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)) # (40, 40, 16)
        self.model.add_module('activation1', nn.ReLU())
        self.model.add_module('pool1', nn.MaxPool2d(kernel_size=2)) # (20, 20, 16)

        self.model.add_module('conv2', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)) # (20, 20, 32)
        self.model.add_module('activation2', nn.ReLU())
        self.model.add_module('pool2', nn.MaxPool2d(kernel_size=2)) # (10, 10, 32)

        self.model.add_module('flatten', nn.Flatten())

        self.model.add_module('linear1', nn.Linear(in_features=3200, out_features=1600))
        self.model.add_module('activation3', nn.ReLU())

        self.model.add_module('linear2', nn.Linear(in_features=1600, out_features=128))
        self.model.add_module('activation3', nn.ReLU())

        self.model.add_module('linear2', nn.Linear(in_features=1600, out_features=10))
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

    # Tensorboard
    writer = SummaryWriter('runs/cnn_shifted_mnist_experiment_' + str(num_epochs) + "_" + str(num_exp))

    # Data Parallelism for Multiple GPU
    dataParallel = DataParallel()
    # setup the process groups
    dataParallel.setup(rank, world_size)
    # Set up the data loader
    train_loader = dataParallel.prepare(True, rank, world_size, batch_size)

    ## Load Full Test data for evaluation
    test_loader = torch.utils.data.DataLoader(DatasetHelper.getDataSet(False), batch_size=batch_size)
    
    print("Training dataset size: ", train_loader.dataset.data.size(0))
    print("Test dataset size: ", test_loader.dataset.data.size(0))

    # Set up the network and optimizer
    network = ShiftedMnistCNN()
    network.to(rank)
    network= DDP(network, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    print(helper.count_parameters(network))
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

            # Convert mnist to shifted MNIST
            data = helper.transformData(data, batch_size)

            data = data.to(rank)
            target = target.to(rank)

            # Get the predictions
            preds = network.forward(data)
            #print(preds.shape)

            # Compute the loss
            loss = helper.cost(preds, target)

            # Take a gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # LR decay
        lr_scheduler.step()
        
        # Calculate accuracies on whole dataset
        train_accuracy, test_accuracy, train_loss, test_loss= helper.evaluate(network, epoch, batch_size, writer, rank, True)
        
        print('Epoch:', '{:3d}'.format(epoch + 1),
              '\tTraining Accuracy:', '{:10.5f}'.format(train_accuracy),
              '\tTraining Loss:', '{:10.5f}'.format(train_loss),
              '\tTesting Accuracy:', '{:10.5f}'.format(test_accuracy),
              '\tTesting Loss:', '{:10.5f}'.format(test_loss))

        # Log LR
        writer.add_scalar('learning rate', lr_scheduler.get_last_lr()[0], (epoch + 1))
        # log the training loss
        writer.add_scalar('training epoch loss', train_loss, (epoch+1))
        # log the evaluation loss
        writer.add_scalar('evaluation epoch loss', test_loss, (epoch+1))
        # log accuracies
        writer.add_scalar('Training epoch Accuracy', train_accuracy, (epoch+1))
        writer.add_scalar('Testing epoch Accuracy', test_accuracy, (epoch+1))
        
        train_acc_l.append(train_accuracy)
        test_acc_l.append(test_accuracy)
        
        if test_accuracy == max(test_acc_l):
            best_epoch = epoch + 1

            # Saving the model with best test accuracy till current epoch
            torch.save(network.state_dict(), model_path + "cnn_shifted_mnist_" + str(num_epochs) + "_" + str(epoch+1) + ".pt")


    writer.flush()
    writer.close()


    # Display Max Accuracy
    print(" Max Train Accuracy : ", max(train_acc_l))
    print(" Max Test Accuracy : ", max(test_acc_l))
    print(" Best Test Accuracy epoch: ", best_epoch)
    
    
    dataParallel.cleanup()
    
import torch.multiprocessing as mp
if __name__ == '__main__':
    
    # Control variables
    batch_size = int(sys.argv[1])
    num_epochs = int(sys.argv[2])
    learning_rate = 1e-3
    num_exp = int(sys.argv[3])
    model_path = "saved_model/cnn_shifted_mnist/"
    
    # Put no. of GPU's used
    world_size = 2
    mp.spawn(
        main,
        args=(world_size, batch_size,num_epochs,learning_rate, model_path, num_exp),
        nprocs=world_size
    )
    