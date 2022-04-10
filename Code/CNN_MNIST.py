#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:27:08 2022

@author: saksham
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
import torchvision.utils as tvutils

from torch.utils.tensorboard import SummaryWriter

from Modules import Squash, Routing, MarginLoss, Helper

class MnistCNN(nn.Module):

    def __init__(self):
        super(MnistCNN, self).__init__()

        # Define the architecture
        self.model = nn.Sequential()

        self.model.add_module('conv1', nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)) # (28, 28, 16)
        self.model.add_module('activation1', nn.ReLU())
        self.model.add_module('pool1', nn.MaxPool2d(kernel_size=2))

        self.model.add_module('conv2', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)) # (14, 14, 32)
        self.model.add_module('activation2', nn.ReLU())
        self.model.add_module('pool2', nn.MaxPool2d(kernel_size=2)) # (7, 7, 32)

        self.model.add_module('flatten', nn.Flatten())

        self.model.add_module('linear1', nn.Linear(in_features=1568, out_features=128))
        self.model.add_module('activation3', nn.ReLU())

        self.model.add_module('linear2', nn.Linear(in_features=128, out_features=10))
        self.model.add_module('activation4', nn.LogSoftmax(dim=1))

        # Define the loss
        self.loss = nn.NLLLoss()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        :param images: (batch_size, height, width) Images to be classified
        :return predictions: (batch_size, 10) Output predictions (log probabilities)
        """
        return self.model(images)

    def cost(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param predictions: (batch_size, 10) Output predictions (log probabilities)
        :param targets: (batch_size, 1) Target classes
        :return cost: The negative log-likelihood loss
        """
        return self.loss(predictions, targets.view(-1))
    
    
if __name__ == '__main__':
    
    helper = Helper()

    if torch.cuda.is_available():
        print("GPU available")
        dev = "cuda:0"
    else:
        dev = "cpu"

    # Control variables
    batch_size = int(sys.argv[1])
    num_epochs = int(sys.argv[2])
    learning_rate = 1e-3

    # Tensorboard
    writer = SummaryWriter('runs/capsule_mnist_experiment_1')

    # Set up the data loader
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./Data/mnist/', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./Data/mnist', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size)

    print("Training dataset size: ", train_loader.dataset.data.size(0))
    print("Test dataset size: ", test_loader.dataset.data.size(0))

    # Set up the network and optimizer
    network = MnistCNN()
    network.to(torch.device(dev))
    print(helper.count_parameters(network))
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

    # Train the network
    for epoch in range(num_epochs):

        train_running_loss = 0.0
        test_running_loss = 0.0

        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            
            data = data.to(torch.device(dev))
            target = target.to(torch.device(dev))

            # Get the predictions
            caps, reconstructions, preds = network.forward(data)
            print(preds.shape)

            # Compute the loss
            loss = network.cost(caps, target, reconstructions, data, True)

            # Take a gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Show the loss
            print('Epoch:', '{:3d}'.format(epoch + 1),
                  '\tTraining Batch:', '{:3d}'.format(batch_idx + 1),
                  '\tTraining Loss:', '{:10.5f}'.format(loss.item()/ data.size(0)))

            train_running_loss += loss.item()

        # LR decay
        lr_scheduler.step()

        # epoch loss
        epoch_loss = train_running_loss / train_loader.dataset.data.size(0)
        # ...log the training loss, lr
        writer.add_scalar('training epoch loss', epoch_loss, (epoch+1))
        writer.add_scalar('learning rate', lr_scheduler.get_last_lr()[0], (epoch + 1))

        # visualize training image reconstruction
        grid = tvutils.make_grid(reconstructions)
        writer.add_image('train_images', grid, epoch+1)


        ## For every epoch calculate validation/testing loss
        network.eval()
        for batch_idx, (data, target) in enumerate(test_loader):
            
            data = data.to(torch.device(dev))
            target = target.to(torch.device(dev))

            caps, reconstructions, preds = network.forward(data)

            batch_loss = network.cost(caps, target, reconstructions, data, True)
            print('Epoch:', '{:3d}'.format(epoch + 1),
                  '\tTesting Batch:', '{:3d}'.format(batch_idx + 1),
                  '\tTesting Loss:', '{:10.5f}'.format(batch_loss.item()/ data.size(0)))

            test_running_loss += batch_loss.item()

        # epoch loss
        epoch_loss = test_running_loss / test_loader.dataset.data.size(0)
        # ...log the evaluation loss
        writer.add_scalar('evaluation epoch loss', epoch_loss, (epoch+1))

        # visualize validation image reconstruction
        grid = tvutils.make_grid(reconstructions)
        writer.add_image('val_images', grid, epoch + 1)

    
    network.eval()
    # Compute accuracy on training set
    count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data = data.to(torch.device(dev))
        target = target.to(torch.device(dev))
        
        _, _, preds = network.forward(data)
        count += torch.sum(preds == target).detach().item()
    print('Training Accuracy:', float(count) / train_loader.dataset.data.size(0))

    # Compute accuracy on test set
    count = 0
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(test_loader):
        
        data = data.to(torch.device(dev))
        target = target.to(torch.device(dev))
        
        caps, reconstructions, preds = network.forward(data)
        count += torch.sum(preds == target).detach().item()
        
        batch_loss = network.cost(caps, target, reconstructions, data, True)
        print('Test Batch Loss:', batch_loss.item()/ data.size(0) )
        running_loss += batch_loss.item()
        
    total_loss_wr = running_loss / test_loader.dataset.data.size(0)

    print('Test Accuracy:', float(count) / test_loader.dataset.data.size(0))
    print('Test Loss:', total_loss_wr)


    writer.flush()
    writer.close()


    # Saving the model
    torch.save(network, "caps_net_mnist_" + num_epochs +".pt")
    
    
    
    