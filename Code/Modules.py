#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 20:36:13 2022

@author: saksham
"""

import torch
from torch import linalg as LA
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import torch.nn as nn

from prettytable import PrettyTable

class Squash():
    
    # Use if any parameter required like epsilon
    ##def __init__():
        
    def perform(self, s: torch.Tensor):
        
        #s_norm = LA.norm(s, dim=-1)
        
        #return ((s_norm ** 2)/(1 + s_norm **2)) * s / s_norm
        
        
        s2 = (s ** 2).sum(dim=-1, keepdims=True)
        
        return (s2 / (1 + s2)) * (s / torch.sqrt(s2))
    
class Routing(nn.Module):
    
    def __init__(self, in_caps: int, out_caps: int, in_d: int, out_d: int, iterations: int):
        
        super(Routing,self).__init__()

        self.in_caps = in_caps
        self.out_caps = out_caps
        self.iterations = iterations
        self.softmax = nn.Softmax(dim=1)
        self.squash = Squash()
        
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

class Helper():


    def count_parameters(model):
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