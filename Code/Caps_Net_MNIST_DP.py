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

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.tensorboard import SummaryWriter

from Modules import Squash, Routing, Helper, DataParallel

class MNISTCapsuleNetworkModel(nn.Module):
    #TODO take dynamic parameters for routing, input size etc
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
    
def main(rank, world_size, batch_size, num_epochs, learning_rate, model_path):
    
    print(rank, world_size, batch_size, num_epochs, learning_rate, model_path)
    
    helper = Helper()

    # Tensorboard
    writer = SummaryWriter('runs/capsule_mnist_experiment_1')

    # Data Parallelism for Multiple GPU
    dataParallel = DataParallel()
    # setup the process groups
    dataParallel.setup(rank, world_size)
    # Set up the data loader
    train_loader = dataParallel.prepare(True, rank, world_size, batch_size)

    test_loader = dataParallel.prepare(False, rank, world_size, batch_size)
    
    print("Training dataset size: ", train_loader.dataset.data.size(0))
    print("Test dataset size: ", test_loader.dataset.data.size(0))

    # Set up the network and optimizer
    network = MNISTCapsuleNetworkModel()
    network.to(rank)
    network= DDP(network, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    print(helper.count_parameters(network))
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    # Train the network
    for epoch in range(num_epochs):

        train_running_loss = 0.0
        test_running_loss = 0.0
        
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)

        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            
            data = data.to(rank)
            target = target.to(rank)

            # Get the predictions
            caps, reconstructions, preds = network.forward(data)
            print(preds.shape)

            # Compute the loss
            loss = helper.cost(caps, target, reconstructions, data, True)

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
        # log the training loss
        writer.add_scalar('training epoch loss', epoch_loss, (epoch+1))

        ## For every epoch calculate validation/testing loss
        network.eval()
        for batch_idx, (data, target) in enumerate(test_loader):
            
            data = data.to(rank)
            target = target.to(rank)

            caps, reconstructions, preds = network.forward(data)

            batch_loss = helper.cost(caps, target, reconstructions, data, True)
            print('Epoch:', '{:3d}'.format(epoch + 1),
                  '\tTesting Batch:', '{:3d}'.format(batch_idx + 1),
                  '\tTesting Loss:', '{:10.5f}'.format(batch_loss.item()/ data.size(0)))

            test_running_loss += batch_loss.item()

        # epoch loss
        epoch_loss = test_running_loss / test_loader.dataset.data.size(0)
        #log the evaluation loss
        writer.add_scalar('evaluation epoch loss', epoch_loss, (epoch+1))


    writer.flush()
    writer.close()


    # Saving the model
    torch.save(network.state_dict(), model_path)
    
    dataParallel.cleanup()
    
import torch.multiprocessing as mp
from collections import OrderedDict
import re
if __name__ == '__main__':
    
    # Control variables
    batch_size = int(sys.argv[1])
    num_epochs = int(sys.argv[2])
    learning_rate = 1e-3
    model_path = "caps_net_mnist_" + str(num_epochs) +".pt"
    
    # Put no. of GPU's used
    world_size = 2
    mp.spawn(
        main,
        args=(world_size, batch_size,num_epochs,learning_rate, model_path),
        nprocs=world_size
    )
    
    helper = Helper()
    
    # When using DDP, state dict add module prefix to all parameters
    # Remove that to load model in non DDP
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    state_dict = torch.load(model_path)
    for k,v in state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict = state_dict
    
    # Use loaded model to evaluate
    loaded_network = MNISTCapsuleNetworkModel()
    loaded_network.load_state_dict(model_dict)

    helper.evaluate(loaded_network, batch_size)
    