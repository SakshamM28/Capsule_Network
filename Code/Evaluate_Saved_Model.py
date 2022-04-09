#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 14:43:25 2022

@author: saksham
"""

from collections import OrderedDict
import re
import sys

import torch.utils.data
from Modules import DatasetHelper
from Caps_Net_MNIST_DP import MNISTCapsuleNetworkModel

from affNIST import getDataset

def evaluate(network, test_loader, batch_size):
    if torch.cuda.is_available():
        print("GPU available")
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
        
    network.to(dev)
    network.eval()
    
    count = 0
    
    for batch_idx, (data, target) in enumerate(test_loader):
        
        print(data.size())
        print(target.size())
        
        data = data.to(dev)
        target = target.to(dev)
        
        _,_, preds = network.forward(data)
        count += torch.sum(preds == target).detach().item()
        
    accuracy = float(count) / len(test_loader.dataset)
    
    print("Accuracy : ", accuracy)


if __name__ == '__main__':
    
    # Control variables
    batch_size = int(sys.argv[1])

    #model_path = "saved_model/250_e/caps_net_mnist_250.pt"
    model_path = "/Users/saksham/Desktop/IISc/Jan 22/E0 270-O ML/Project/GPU_results/250_e/caps_net_mnist_250.pt"
    
    # Get required dataset
    test_loader = torch.utils.data.DataLoader(DatasetHelper.getDataSet(False), batch_size=batch_size)
    
    #test_loader = torch.utils.data.DataLoader(getDataset(), batch_size=batch_size)
    
    # When using DDP, state dict add module prefix to all parameters
    # Remove that to load model in non DDP
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    for k,v in state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict = state_dict
    
    # Use loaded model to evaluate
    loaded_network = MNISTCapsuleNetworkModel()
    loaded_network.load_state_dict(model_dict)

    evaluate(loaded_network, test_loader, batch_size)