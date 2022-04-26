#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 19:08:24 2022

@author: saksham
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Modules_Caps_Text import Helper, DatasetHelper, Routing, Squash
from torch.utils.tensorboard import SummaryWriter

class ImdbCapsuleNetworkModel(nn.Module):

    def __init__(self, embedding_dim, max_words, n_filters, filter_size, output_dim, dropout):
      '''
      params
      embedding_dim : length of embedding used
      n_filters : depth of hidden layers
      filter_sizes : list of filter sizes
      output_dim : number of classes
      dropout : dropout value
      '''
      super(ImdbCapsuleNetworkModel, self).__init__()

      self.conv1 = nn.Conv2d(in_channels = 1, out_channels = n_filters, kernel_size = [filter_size, embedding_dim], stride=1)
      self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=32 * 8, kernel_size=[filter_size,1], stride=1, padding=0)
      
      self.squash = Squash()
      #TODO Check outputs after connvs
      self.conv_out = max_words - 2*filter_size + 2
      
      self.text_capsules = Routing(32 * self.conv_out , output_dim , 8, 16, 3)
      
      # TODO Check dropout on layers 
      self.dropout = nn.Dropout(dropout)

    def forward(self, text):
            
      #text = [batch size, sent len, emb dim]
      print(text.shape)

      
      embedded = text.unsqueeze(1)
      #embedded = [batch size, 1, sent len, emb dim]
      print(embedded.shape)
      
      x = F.relu(self.conv1(embedded))
      print(x.shape)
      x = self.conv2(x)
      print(x.shape)
      
      caps = x.view(x.shape[0], 8, 32 * self.conv_out).permute(0, 2, 1)
      caps = self.squash.perform(caps)
      caps = self.text_capsules.perform(caps)
      
      
      with torch.no_grad():
          pred = (caps ** 2).sum(-1).argmax(-1)
          
      return caps, pred


def main(batch_size, num_epochs, learning_rate, model_path, num_exp, max_words, embed_len):
    print(batch_size, num_epochs, learning_rate, model_path, num_exp, max_words, embed_len)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model Initialization variables
    N_FILTERS = 100
    FILTER_SIZE = 3
    OUTPUT_DIM = 2
    DROPOUT = 0.5

    helper = Helper(max_words,embed_len, OUTPUT_DIM)

    # Tensorboard
    writer = SummaryWriter('runs/caps_net_imdb_experiment_' + str(num_epochs) + "_" + str(num_exp))

    train_loader = torch.utils.data.DataLoader(DatasetHelper.getIMDBDataSet(True), batch_size=batch_size, shuffle=True, collate_fn=helper.vectorize_batch)
    test_loader  = torch.utils.data.DataLoader(DatasetHelper.getIMDBDataSet(False), batch_size=batch_size, shuffle=True, collate_fn=helper.vectorize_batch)

    print("Training dataset size: ", len(train_loader.dataset))
    print("Test dataset size: ", len(test_loader.dataset))

    network = ImdbCapsuleNetworkModel(embed_len, max_words, N_FILTERS, FILTER_SIZE, OUTPUT_DIM, DROPOUT)
    network.to(device)
    
    print(helper.count_parameters(network))

    # TODO Checl L2 penealty
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

    train_acc_l = []
    test_acc_l = []
    best_epoch = 0
    # Train the network
    for epoch in range(num_epochs):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):

            data = data.to(device)
            target = target.to(device)

            # Get the predictions
            caps, preds = network.forward(data)
            #print(preds.shape)

            # Compute the loss
            loss = helper.cost(caps, target)
            
            # Take a gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # LR decay
        lr_scheduler.step()
        
        # Calculate accuracies on whole dataset
        train_accuracy, test_accuracy, train_loss, test_loss= helper.evaluate(network, epoch, batch_size)
        
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
            torch.save(network.state_dict(), model_path + "caps_net_imdb_" + str(num_epochs) + "_" + str(epoch+1) + ".pt")


    writer.flush()
    writer.close()


    # Display Max Accuracy
    print(" Max Train Accuracy : ", max(train_acc_l))
    print(" Max Test Accuracy : ", max(test_acc_l))
    print(" Best Test Accuracy epoch: ", best_epoch)

import os
if __name__ == '__main__':
    
    # Control variables
    batch_size = int(sys.argv[1])
    num_epochs = int(sys.argv[2])
    learning_rate = 1e-3
    num_exp = int(sys.argv[3])
    model_path = "saved_model/caps_net_imdb/"
    
    if os.path.exists(model_path) == False:
        os.mkdir(model_path)

    #TODO : Preprocess data to find suitable max words per sentence
    max_words = 200
    embed_len = 100
    
    main(batch_size, num_epochs, learning_rate, model_path, num_exp, max_words, embed_len)