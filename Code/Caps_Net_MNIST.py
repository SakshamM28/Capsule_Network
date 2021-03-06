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

from torch.utils.tensorboard import SummaryWriter

from Modules import Squash, Routing, MarginLoss, Helper

class MNISTCapsuleNetworkModel(nn.Module):
    
    def __init__(self):
        super(MNISTCapsuleNetworkModel, self).__init__()
        
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=32 * 8, kernel_size=9, stride=2, padding=0)
        
        self.squash = Squash()
        
        self.digit_capsules = Routing(32 * 6 * 6, 10, 8, 16, 3)
        
        
        self.decoder = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )
        
        self.margin_loss = MarginLoss(n_labels=10)
        self.reconstruction_loss = nn.MSELoss(reduction='sum')
        
    def forward(self, data: torch.Tensor):
        
        x = F.relu(self.conv1(data))
        x = self.conv2(x)
        
        caps = x.view(x.shape[0], 8, 32 * 6 * 6).permute(0, 2, 1)
        caps = self.squash.perform(caps)
        caps = self.digit_capsules.perform(caps)
        
        
        with torch.no_grad():
            pred = (caps ** 2).sum(-1).argmax(-1)
            mask = torch.eye(10, device=data.device)[pred]
            
        reconstructions = self.decoder((caps * mask[:, :, None]).view(x.shape[0], -1))
        
        reconstructions = reconstructions.view(-1, 1, 28, 28)
        
        return caps, reconstructions, pred
    
    def cost(self, caps: torch.Tensor, targets: torch.Tensor, reconstructions: torch.Tensor, data: torch.Tensor, isReconstruction = False) -> torch.Tensor:
        
        margin_loss = self.margin_loss.calculate(caps, targets)
        if isReconstruction == True:
            return  margin_loss + 0.0005 * self.reconstruction_loss(reconstructions, data)
        
        return margin_loss
    
    
if __name__ == '__main__':
    
    helper = Helper()
    
    if torch.cuda.is_available():
        print("GPU available")
        dev = "cuda"
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
    network = MNISTCapsuleNetworkModel()
    network.to(torch.device(dev))
    print(helper.count_parameters(network))
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

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

        # epoch loss
        epoch_loss = train_running_loss / train_loader.dataset.data.size(0)
        # ...log the training loss
        writer.add_scalar('training epoch loss', epoch_loss, (epoch+1))

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
    
    
    
    