#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 20:36:13 2022

@author: saksham
"""

import os
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from torchvision import datasets, transforms
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from prettytable import PrettyTable

class Squash():
    
    # Use if any parameter required like epsilon
    ##def __init__():
    #TODO Use some epsillon in divison and check performance
    def perform(self, s: torch.Tensor):
        
        s2 = (s ** 2).sum(dim=-1, keepdims=True)
        
        return (s2 / (1 + s2)) * (s / torch.sqrt(s2))
    
class Routing(nn.Module):
    
    def __init__(self, in_caps: int, out_caps: int, in_d: int, out_d: int, iterations: int):
        
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
        
        u_hat = torch.einsum('ijnm,bin->bijm', self.weight, u)
        
        b = u.new_zeros(u.shape[0], self.in_caps, self.out_caps)
        
        v = None
        
        for i in range(self.iterations):
            
            c = self.softmax(b)
            s = torch.einsum('bij,bijm->bjm', c, u_hat)
            v = self.squash.perform(s)
            # agreement
            a = torch.einsum('bjm,bijm->bij', v, u_hat)
            
            b = b + a
            
        return v
    
class MarginLoss():
    
    def __init__(self, *, n_labels: int, lambda_: float = 0.5, m_positive: float = 0.9, m_negative: float = 0.1):
        
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
    def getDataSet(isTrain):
        
        return datasets.MNIST('./Data/mnist/', train=isTrain, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        

class DataParallel():
    
    def setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
    def prepare(self, isTrain, rank, world_size, batch_size=128, pin_memory=False, num_workers=0):
        dataset = DatasetHelper.getDataSet(isTrain)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    
        return dataloader

    def cleanup(self):
        dist.destroy_process_group()

class Helper():
    
    def __init__(self):
        
        self.margin_loss = MarginLoss(n_labels=10)
        self.reconstruction_loss = nn.MSELoss(reduction='sum')


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
    
    
    def cost(self, caps: torch.Tensor, targets: torch.Tensor, reconstructions: torch.Tensor, data: torch.Tensor, isReconstruction = False) -> torch.Tensor:
        
        margin_loss = self.margin_loss.calculate(caps, targets)
        if isReconstruction == True:
            return  margin_loss + 0.0005 * self.reconstruction_loss(reconstructions, data)
        
        return margin_loss
    
    def evaluate(self, network, batch_size):
        
        if torch.cuda.is_available():
            print("GPU available")
            dev = "cuda"
        else:
            dev = "cpu"
        
        train_loader = torch.utils.data.DataLoader(DatasetHelper.getDataSet(True), batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(DatasetHelper.getDataSet(False), batch_size=batch_size)
        
        network.to(torch.device(dev))
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
            
            batch_loss = self.cost(caps, target, reconstructions, data, True)
            print('Test Batch Loss:', batch_loss.item()/ data.size(0) )
            running_loss += batch_loss.item()
            
        total_loss_wr = running_loss / test_loader.dataset.data.size(0)

        print('Test Accuracy:', float(count) / test_loader.dataset.data.size(0))
        print('Test Loss:', total_loss_wr)