#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:35:34 2022

@author: saksham
"""

import os
import torch
import torch.nn as nn
from torchtext import datasets
from torchtext.data.utils import get_tokenizer

import torch.nn.functional as F

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from torchtext.data.functional import to_map_style_dataset
from torchtext import vocab

from prettytable import PrettyTable

class Squash():
    '''
    Like a activation function but performed on whole layer output and not neuron/element wise
    '''
    
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
    def perform(self, s: torch.Tensor):
        
        s2 = (s ** 2).sum(dim=-1, keepdims=True)
        
        # Adding epsilon in case s2 become 0
        return (s2 / (1 + s2)) * (s / torch.sqrt(s2 + self.epsilon))

class Routing(nn.Module):
    
    def __init__(self, in_caps: int, out_caps: int, in_d: int, out_d: int, iterations: int):

        '''
        params
        in_caps : Input from previvous layer reshaped to form capsules, 32*8 channels divided in 2 parts
        32 as number of capsules and 8 as input dimension

        out_caps : 2 -  1 for every class

        in_d : imput dimension (8)

        out_d : output dimension (length of each vector of output class)

        iterations : routing loop iterations
        '''

        super(Routing,self).__init__()

        #Intitialization with parameters required for routing
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.iterations = iterations
        self.softmax = nn.Softmax(dim=1)
        self.squash = Squash()

        # Weight matrix learned during training
        self.weight = nn.Parameter(torch.randn(in_caps, out_caps, in_d, out_d), requires_grad=True)


    def perform(self, u):

        self.weight = self.weight.to(u.device)

        # Prediction Vectors (Sumed over input dimentions)
        u_hat = torch.einsum('ijnm,bin->bijm', self.weight, u)

        b = u.new_zeros(u.shape[0], self.in_caps, self.out_caps)

        v = None

        for i in range(self.iterations):

            c = self.softmax(b)
            # Weighted sum over pridiction vectors
            s = torch.einsum('bij,bijm->bjm', c, u_hat)
            v = self.squash.perform(s)
            # agreement
            a = torch.einsum('bjm,bijm->bij', v, u_hat)

            b = b + a

        return v

class MarginLoss():
    
    def __init__(self, *, n_labels: int, lambda_: float = 0.5, m_positive: float = 0.9, m_negative: float = 0.1):

        '''
        params

        n_labels : Totals classes
        lambda_ : Use to decrease loss if wrong result so that training is not affected in start
        m_positive : weight of positive loss
        m_negative : weight of negative loss
        '''

        self.m_negative = m_negative
        self.m_positive = m_positive
        self.lambda_ = lambda_
        self.n_labels = n_labels

    def calculate(self, v: torch.Tensor, labels: torch.Tensor):

        v_norm = torch.sqrt((v ** 2).sum(dim=-1))

        labels = torch.eye(self.n_labels, device=labels.device)[labels]

        loss = labels * F.relu(self.m_positive - v_norm) + self.lambda_ * (1.0 - labels) * F.relu(v_norm - self.m_negative)

        return loss.sum(dim=-1).sum()

class DatasetHelper():
    def getIMDBDataSet(isTrain):
        if isTrain:
            split = 'train'
        else:
            split = 'test'
        
        #return to_map_style_dataset(datasets.IMDB(root='./Data/IMDB/' ,split = split) )
        return to_map_style_dataset(datasets.IMDB(root='/scratch/sakshamgoyal/' ,split = split) )
    
class DataPreProcess():
    def __init__(self, max_words, embed_len):
        '''
        param max_words : len of single sample/document
        param embed_len : length of embegging vector
        '''
        
        self.tokenizer = get_tokenizer("basic_english")
        self.max_words = max_words
        self.embed_len = embed_len
        
        # Load Pretrained GloVe embeddings
        #self.glove = vocab.GloVe(name='6B', dim=self.embed_len, cache = './Data/GloVE/')
        #self.glove = vocab.GloVe(name='6B', dim=self.embed_len, cache = '/scratch/sakshamgoyal/')
        self.glove = vocab.GloVe(name='840B', dim=self.embed_len, cache = '/scratch/sakshamgoyal/')

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

class DataParallel():

    def setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '13445'
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    def prepare(self, isTrain, rank, world_size, batch_size=128, pin_memory=False, num_workers=0, pre_process=None):
        dataset = DatasetHelper.getIMDBDataSet(isTrain)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler, collate_fn=pre_process.vectorize_batch)

        return dataloader

    def cleanup(self):
        dist.destroy_process_group()

class Helper():

    def __init__(self, n_labels, pre_process):
        '''
        param n_labels : Number of output classes
        '''
        self.margin_loss = MarginLoss(n_labels=n_labels)
        self.loss = nn.NLLLoss(reduction='sum')

        self.pre_process = pre_process

    def cost(self, caps: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        margin_loss = self.margin_loss.calculate(caps, targets)
        return margin_loss

    def count_parameters(self, model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        #print(table)
        #print(f"Total Trainable Params: {total_params}")
        return table, total_params

    def evaluate(self, network, epoch, batch_size, rank = None):

        if rank:
            dev = rank
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
            if rank == 0:
                print("GPU available")
        else:
            dev = torch.device("cpu")

        train_loader = torch.utils.data.DataLoader(DatasetHelper.getIMDBDataSet(True), batch_size=batch_size, shuffle=True, collate_fn=self.pre_process.vectorize_batch)
        test_loader  = torch.utils.data.DataLoader(DatasetHelper.getIMDBDataSet(False), batch_size=batch_size, shuffle=True, collate_fn=self.pre_process.vectorize_batch)

        network.to(dev)
        network.eval()
        # Compute accuracy on training set
        count = 0
        train_running_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(train_loader):

                data = data.to(dev)
                target = target.to(dev)

                caps , preds = network.forward(data)
                count += torch.sum(preds == target).detach().item()

                batch_loss = self.cost(caps, target)
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

                caps , preds = network.forward(data)
                count += torch.sum(preds == target).detach().item()

                batch_loss = self.cost(caps, target)
                test_running_loss += batch_loss.item()

        test_loss = test_running_loss / len(test_loader.dataset)

        test_accuracy = float(count) / len(test_loader.dataset)
        print('Test Accuracy:', test_accuracy)
        print('Test Loss:', test_loss)

        return train_accuracy, test_accuracy, train_loss, test_loss
