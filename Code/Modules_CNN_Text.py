#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:58:55 2022

@author: saksham
"""

import torch
import torch.nn as nn
from torchtext import datasets
from torchtext.data.utils import get_tokenizer

from torchtext.data.functional import to_map_style_dataset
from torchtext import vocab

from prettytable import PrettyTable


class DatasetHelper():
    def getIMDBDataSet(isTrain):
        if isTrain:
            split = 'train'
        else:
            split = 'test'
        
        #return to_map_style_dataset(datasets.IMDB(root='./Data/IMDB/' ,split = split) )
        return to_map_style_dataset(datasets.IMDB(root='/scratch/sakshamgoyal/' ,split = split) )

class Helper():
    
    def __init__(self, max_words, embed_len):
        '''
        param max_words : len of single sample/document
        param embed_len : length of embegging vector
        '''
        
        self.tokenizer = get_tokenizer("basic_english")
        self.loss = nn.NLLLoss(reduction='sum')
        self.max_words = max_words
        self.embed_len = embed_len
        
        # Load Pretrained GloVe embeddings
        #self.glove = vocab.GloVe(name='6B', dim=self.embed_len, cache = './Data/GloVE/')
        self.glove = vocab.GloVe(name='6B', dim=self.embed_len, cache = '/scratch/sakshamgoyal/')
        #self.glove = vocab.GloVe(name='840B', dim=self.embed_len, cache = '/scratch/sakshamgoyal/')
        
    def cost(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param predictions: (batch_size, 2) Output predictions (log probabilities)
        :param targets: (batch_size, 1) Target classes
        :return cost: The negative log-likelihood loss
        """
        return self.loss(predictions, targets.view(-1))
    
    def vectorize_batch(self, batch):
        '''
        Gets a batch and return embeddings after padding or truncating to max_words
        param batch : Batch of data
        return embedding of shape [Batch, max_words, embed_len] and targest of shape [Batch]
        '''
        Y, X = list(zip(*batch))
        X = [self.tokenizer(x) for x in X]
        X = [tokens+[""] * (self.max_words-len(tokens))  if len(tokens) < self.max_words else tokens[: self.max_words] for tokens in X]
        X_tensor = torch.zeros(len(batch), self.max_words, self.embed_len)
        for i, tokens in enumerate(X):
            #TODO Lower case lookup can be checked
            X_tensor[i] = self.glove.get_vecs_by_tokens(tokens)
        
        Y = [int(y == 'pos')  for y in Y]
        return X_tensor, torch.tensor(Y)

    def count_parameters(self, model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params
    
    def evaluate(self, network, epoch, batch_size):
        
        if torch.cuda.is_available():
            print("GPU available")
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")
            
        train_loader = torch.utils.data.DataLoader(DatasetHelper.getIMDBDataSet(True), batch_size=batch_size, shuffle=True, collate_fn=self.vectorize_batch)
        test_loader  = torch.utils.data.DataLoader(DatasetHelper.getIMDBDataSet(False), batch_size=batch_size, shuffle=True, collate_fn=self.vectorize_batch)
        
        network.to(dev)
        network.eval()
        # Compute accuracy on training set
        count = 0
        train_running_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(train_loader):

                data = data.to(dev)
                target = target.to(dev)

                preds = network.forward(data)
                _, pred = torch.max(preds, dim=1)
                count += torch.sum(pred == target).detach().item()

                batch_loss = self.cost(preds, target)

                train_running_loss += batch_loss.item()

        train_loss = train_running_loss / len(train_loader.dataset)

        train_accuracy = float(count) / len(train_loader.dataset)

        # Compute accuracy on test set
        count = 0
        test_running_loss = 0.0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):

                data = data.to(dev)
                target = target.to(dev)

                preds = network.forward(data)
                _, pred = torch.max(preds, dim=1)
                count += torch.sum(pred == target).detach().item()

                batch_loss = self.cost(preds, target)
                #print('Test Batch Loss:', batch_loss.item()/ data.size(0) )
                test_running_loss += batch_loss.item()

        test_loss = test_running_loss / len(test_loader.dataset)

        test_accuracy = float(count) / len(test_loader.dataset)
        print('Test Accuracy:', test_accuracy)
        print('Test Loss:', test_loss)

        return train_accuracy, test_accuracy, train_loss, test_loss
