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
from Modules import Helper
from Caps_Net_ShiftedMNIST_DP import ShiftedMNISTCapsuleNetworkModel
from Caps_Net_MNIST_DP import MNISTCapsuleNetworkModel
from CNN_MNIST_DP import MnistCNN
from CNN_ShiftedMNIST_DP import ShiftedMnistCNN

from affNIST import getDataset

def evaluate(network, test_loader, batch_size, model_arch):
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
        
        if model_arch == 1 or model_arch == 3: # Caps Net
            _,_, preds = network.forward(data)
            count += torch.sum(preds == target).detach().item()

        elif model_arch == 2 or model_arch == 4: # CNN
            preds = network.forward(data)
            _, pred = torch.max(preds, dim=1)
            count += torch.sum(pred == target).detach().item()
        
    accuracy = float(count) / len(test_loader.dataset)
    
    print("Accuracy : ", accuracy)


if __name__ == '__main__':
    
    '''
    Agruments Required
    1. Batch Size
    2. Model Architecture {1 : Caps_Net_Mnist, 2 : CNN_Mnist, 3 : Caps_Net_ShiftetMnist, 4 : CNN_ShiftedMnist}
    '''
    # Control variables
    batch_size = int(sys.argv[1])
    model_arch = int(sys.argv[2])

    isResized = False

    if model_arch == 1:
        # Caps Net MNIST model
        model_path = "saved_model/best_models/caps_mnist/caps_net_mnist_250_75.pt"
    elif model_arch == 2:
        # CNN MNIST model
        model_path = "saved_model/best_models/cnn_mnist/cnn_mnist_100_100.pt"
    elif model_arch == 3:
        # Caps Net Shifted MNIST model
        model_path = "saved_model/best_models/caps_shifted_mnist/caps_net_shifted_mnist_250_161.pt"
        isResized = False
    elif model_arch == 4:
        # CNN Shifted MNIST model
        model_path = "saved_model/best_models/cnn_shifted_mnist/cnn_shifted_mnist_100_82.pt"
        isResized = False


    #model_path = "/Users/saksham/Desktop/IISc/Jan 22/E0 270-O ML/Project/GPU_results/250_e/caps_net_mnist_250.pt"
    
    # Get required dataset
    # MNIST test data
    #test_loader = torch.utils.data.DataLoader(DatasetHelper.getDataSet(False), batch_size=batch_size)
    
    # affNIST resized test data
    test_loader = torch.utils.data.DataLoader(getDataset(isResized), batch_size=batch_size)
    
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

    if model_arch == 1:
        # Caps Net MNIST model
        loaded_network = MNISTCapsuleNetworkModel()
    elif model_arch == 2:
        # CNN MNIST model
        loaded_network = MnistCNN()
    elif model_arch == 3:
        # Caps Net Shifted MNIST model
        loaded_network = ShiftedMNISTCapsuleNetworkModel()
    elif model_arch == 4:
        # CNN Shifted MNIST model
        loaded_network = ShiftedMnistCNN()

    helper = Helper()
    print(helper.count_parameters(loaded_network))

    loaded_network.load_state_dict(model_dict)

    evaluate(loaded_network, test_loader, batch_size, model_arch)