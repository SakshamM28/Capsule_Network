#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:59:42 2022

@author: saksham
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Modules_CNN_Text import Helper, DatasetHelper
from torch.utils.tensorboard import SummaryWriter

class ImdbCNN1d(nn.Module):

    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
      '''
      params
      embedding_dim : length of embedding used
      n_filters : depth of hidden layers
      filter_sizes : list of filter sizes
      output_dim : number of classes
      dropout : dropout value
      '''
      super(ImdbCNN1d, self).__init__()

      self.convs = nn.ModuleList([
                                  nn.Conv1d(in_channels = embedding_dim, 
                                            out_channels = n_filters, 
                                            kernel_size = fs)
                                  for fs in filter_sizes
                                  ])

      self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

      self.dropout = nn.Dropout(dropout)

      self.class_activation = nn.LogSoftmax(dim = 1)

    def forward(self, text):
            
      #text = [batch size, sent len, emb dim]
      
      embedded = text.permute(0, 2, 1)
      
      #embedded = [batch size, emb dim, sent len]
      
      conved = [F.relu(conv(embedded)) for conv in self.convs]
          
      #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
      
      # Max Pool over length on sentence (direction of stride)
      pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
      
      #pooled_n = [batch size, n_filters]
      
      cat = self.dropout(torch.cat(pooled, dim = 1))
      
      #cat = [batch size, n_filters * len(filter_sizes)]
          
      return self.class_activation(self.fc(cat))


def main(batch_size, num_epochs, learning_rate, model_path, num_exp, max_words, embed_len, l2_penalty):
    print(batch_size, num_epochs, learning_rate, model_path, num_exp, max_words, embed_len, l2_penalty)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    helper = Helper(max_words,embed_len)

    # Tensorboard
    writer = SummaryWriter('runs/cnn_imdb_experiment_' + str(num_epochs) + "_" + str(num_exp))

    train_loader = torch.utils.data.DataLoader(DatasetHelper.getIMDBDataSet(True), batch_size=batch_size, shuffle=True, collate_fn=helper.vectorize_batch)
    test_loader  = torch.utils.data.DataLoader(DatasetHelper.getIMDBDataSet(False), batch_size=batch_size, shuffle=True, collate_fn=helper.vectorize_batch)

    print("Training dataset size: ", len(train_loader.dataset))
    print("Test dataset size: ", len(test_loader.dataset))

    # Model Initialization variables
    N_FILTERS = 256
    FILTER_SIZES = [3,4,5]
    OUTPUT_DIM = 2
    DROPOUT = 0.5

    network = ImdbCNN1d(embed_len, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
    network.to(device)
    
    print(helper.count_parameters(network))

    optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay = l2_penalty)
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
            torch.save(network.state_dict(), model_path + "cnn_imdb_" + str(num_epochs) + "_" + str(epoch+1) + ".pt")


    writer.flush()
    writer.close()


    # Display Max Accuracy
    print(" Max Train Accuracy : ", max(train_acc_l))
    print(" Max Test Accuracy : ", max(test_acc_l))
    print(" Best Test Accuracy epoch: ", best_epoch)

from pathlib import Path
if __name__ == '__main__':
    
    # Control variables
    batch_size = int(sys.argv[1])
    num_epochs = int(sys.argv[2])
    learning_rate = 1e-4
    num_exp = int(sys.argv[3])
    l2_penalty = float(sys.argv[4])
    model_path = "saved_model/cnn_imdb_" + str(num_exp) + "/"
    
    Path(model_path).mkdir(parents=True, exist_ok=True)

    #TODO : Preprocess data to find suitable max words per sentence
    max_words = 200
    embed_len = 300
    
    main(batch_size, num_epochs, learning_rate, model_path, num_exp, max_words, embed_len, l2_penalty)

