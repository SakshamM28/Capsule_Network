#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:37:52 2022

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

from Modules_Caps_Text import Squash, Routing, Helper, DataParallel, DatasetHelper, DataPreProcess

class ImdbCapsuleNetworkModel(nn.Module):

    def __init__(self, embedding_dim, max_words, n_capsules, n_filters, filter_size, output_dim, dropout):
      '''
      params
      embedding_dim : length of embedding used
      n_filters : depth of hidden layers
      filter_sizes : list of filter sizes
      output_dim : number of classes
      dropout : dropout value
      '''
      super(ImdbCapsuleNetworkModel, self).__init__()

      self.elu_l = nn.Conv2d(in_channels = 1, out_channels = n_filters, kernel_size = [filter_size, embedding_dim], stride=1)
      self.conv1 = nn.Conv2d(in_channels = 1, out_channels = n_filters, kernel_size = [filter_size, embedding_dim], stride=1)
      self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels= n_capsules * 8, kernel_size=[filter_size,1], stride=1, padding=0)
      
      self.squash = Squash()
      #TODO Check outputs after connvs
      self.conv_out = max_words - 2*filter_size + 2
      self.n_capsules = n_capsules
      
      self.text_capsules = Routing(n_capsules * self.conv_out , output_dim , 8, 16, 3)

      self.dropout = nn.Dropout(dropout)

    def forward(self, text):
            
      #text = [batch size, sent len, emb dim]
      #print(text.shape)

      
      embedded = text.unsqueeze(1)
      #embedded = [batch size, 1, sent len, emb dim]

      x_elu = F.elu(self.elu_l(embedded))
      x = F.relu(self.conv1(embedded))

      x = x_elu * x
      x = self.dropout(x)
      x = self.conv2(x)

      caps = x.view(x.shape[0], 8, self.n_capsules * self.conv_out).permute(0, 2, 1)
      # Trying relu to avoid vanishing gradient
      caps = F.relu(caps)
      #caps = self.squash.perform(caps)

      caps = self.dropout(caps)

      caps = self.text_capsules.perform(caps)
      
      with torch.no_grad():
          pred = (caps ** 2).sum(-1).argmax(-1)
          
      return caps, pred

def main(rank, world_size, batch_size, num_epochs, learning_rate, model_path, num_exp, max_words, embed_len, l2_penalty):
    print(rank, world_size, batch_size, num_epochs, learning_rate, model_path, num_exp, max_words, embed_len, l2_penalty)
    
    # Model Initialization variables
    N_CAPSULES = 6
    N_FILTERS = 256
    FILTER_SIZE = 6
    OUTPUT_DIM = 2
    DROPOUT = 0.5

    dataPreProcess = DataPreProcess(max_words, embed_len)
    helper = Helper(OUTPUT_DIM, dataPreProcess)
    
    if rank == 0:
        # Tensorboard
        writer = SummaryWriter('runs/caps_net_imdb_experiment_' + str(num_epochs) + "_" + str(num_exp)) 

    # Data Parallelism for Multiple GPU
    dataParallel = DataParallel()
    # setup the process groups
    dataParallel.setup(rank, world_size)
    # Set up the data loader
    train_loader = dataParallel.prepare(True, rank, world_size, batch_size=batch_size, pre_process=dataPreProcess)
    
    #Test Loader
    #test_loader  = torch.utils.data.DataLoader(DatasetHelper.getIMDBDataSet(False), batch_size=batch_size, shuffle=True, collate_fn=dataPreProcess.vectorize_batch)
    
    if rank == 0:
        print("Training dataset size: ", len(train_loader.dataset))
        #print("Test dataset size: ", len(test_loader.dataset))
        
    network = ImdbCapsuleNetworkModel(embed_len, max_words, N_CAPSULES, N_FILTERS, FILTER_SIZE, OUTPUT_DIM, DROPOUT)
    network.to(rank)
    network= DDP(network, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    if rank == 0:
        table, total_params = helper.count_parameters(network)
        print(table)
        print('Total trainable parameters: ', total_params)
        
    optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=l2_penalty)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    
    train_acc_l = []
    test_acc_l = []
    best_epoch = 0
    # Train the network
    for epoch in range(num_epochs):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):

            data = data.to(rank)
            target = target.to(rank)

            # Get the predictions
            caps, preds = network.forward(data)
            #print(caps)

            # Compute the loss
            loss = helper.cost(caps, target)
            
            # Take a gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # LR decay after every 2 epochs, original paper applies after ~3 epochs
        if (epoch+1)%2 ==0:
            lr_scheduler.step()
            
        # Calculate accuracies on whole dataset
        #if rank == 0:
        train_accuracy, test_accuracy, train_loss, test_loss= helper.evaluate(network, epoch, batch_size, rank)

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
            torch.save(network.state_dict(), model_path + 'caps_net_imdb_' + str(num_epochs) + '_'+ str(epoch+1) + '.pt')

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
from pathlib import Path

if __name__ == '__main__':
    
    # Control variables
    batch_size = int(sys.argv[1])
    num_epochs = int(sys.argv[2])
    learning_rate = 1e-3
    num_exp = int(sys.argv[3])
    l2_penalty = float(sys.argv[4])

    model_path = "saved_model/caps_net_imdb_" + str(num_exp) + "/"
    Path(model_path).mkdir(parents=True, exist_ok=True)
    
    
    #TODO : Preprocess data to find suitable max words per sentence
    # According to papaer avg len of IMDB data is 213, so taking 200
    max_words = 200
    embed_len = 300
    
    
    # Put no. of GPU's used
    world_size = 2
    mp.spawn(
        main,
        args=(world_size, batch_size, num_epochs, learning_rate, model_path, num_exp, max_words, embed_len, l2_penalty),
        nprocs=world_size
    )