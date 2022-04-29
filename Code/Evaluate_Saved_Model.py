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
from MultiMNIST_Dataloader import MultiMNIST_Dataloader
from Modules import Helper

from Caps_Net_MNIST_DP import MNISTCapsuleNetworkModel
from Caps_Net_ShiftedMNIST_DP import ShiftedMNISTCapsuleNetworkModel
from Caps_Net_MultiMNIST_DP import MultiMNISTCapsuleNetworkModel
from CNN_MNIST_DP import MnistCNN
from CNN_ShiftedMNIST_DP import ShiftedMnistCNN
from CNN_MultiMNIST_DP import MultiMnistCNN

from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as tvutils

from affNIST import getDataset

def evaluate(network, test_loader, batch_size, model_arch, writer, test_data, isResized, isShiftedMNIST=False, isMultiMNIST=False):

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print("GPU available")
    else:
        dev = torch.device("cpu")

    network.to(dev)
    network.eval()

    count = 0
    wrong_count = 1
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):

            print('Test image: ', data.size())
            print('Test label: ', target.size())

            data = data.to(dev)
            target = target.to(dev)

            if model_arch in [1, 2, 3]: # CapsNet
                _,reconstructions, preds = network.forward(data)
                count += torch.sum(preds == target).detach().item()

            elif model_arch in [4, 5, 6]: # CNN
                pred = network.forward(data)
                _, preds = torch.max(pred, dim=1)
                count += torch.sum(preds == target).detach().item()

            wrong_idx = ((preds==target) == False).nonzero(as_tuple=False)
            for idx in wrong_idx.cpu().detach().numpy().reshape(-1):
                print('Wrong index:', idx)
                img = data.__getitem__(idx)
                #pred_label = preds[idx]
                #true_label = target.__getitem__(idx)

                grid = tvutils.make_grid(img)
                writer.add_image('wrong_images_' + str(wrong_count), grid, 0)

                if model_arch in [1, 2, 3]:
                    indices = torch.tensor([idx])
                    indices = indices.to(dev)
                    recon_img = torch.index_select(reconstructions, 0 , indices)

                    grid = tvutils.make_grid(recon_img)
                    writer.add_image('wrong_images_' + str(wrong_count), grid, 1)
                wrong_count += 1

            #  For Testing just 1 batch

    accuracy = float(count) / len(test_loader.dataset)

    print("Accuracy : ", accuracy)
    print("Wrong Count", wrong_count - 1)


if __name__ == '__main__':

    '''
    Agruments Required
    1. Batch Size
    2. Model Architecture {1: Caps_Net_Mnist, 2: Caps_Net_ShiftetMnist, 3: Caps_Net_MultiMNIST, 4: CNN_ShiftedMnist, 5: CNN_MNIST, 6: CNN_MultiMNIST}
    '''
    # Control variables
    batch_size = int(sys.argv[1])
    model_arch = int(sys.argv[2])
    test_data = sys.argv[3]

    isResized = True # for affNIST dataset evaluation

    if model_arch == 1:
        # Caps_Net_MNIST
        model_path = 'saved_model/mnist_128_500/caps_net_mnist_500_109.pt' # latest
    elif model_arch == 2:
        # Caps_Net_ShiftedMNIST
        model_path = 'saved_model/shiftedmnist_256_500/caps_net_shifted_mnist_500_139.pt' # latest
        isResized = False
    elif model_arch == 3:
        # Caps_Net_MultiMNIST
        model_path = 'saved_model/best_models/caps_shifted_mnist/caps_net_shifted_mnist_250_161.pt'
    elif model_arch == 4:
        # CNN_ShiftedMNIST
        model_path = 'saved_model/shiftedmnist_cnn_256_500/cnn_multimnist_500_147.pt' # latest
        isResized = False
    elif model_arch == 5:
        # CNN_MNIST
        model_path = 'saved_model/mnist_cnn_128_500/cnn_multimnist_500_136.pt' #latest
    elif model_arch == 6:
        # CNN_MultiMNIST
        model_path = 'saved_model/best_models/cnn_shifted_mnist/cnn_shifted_mnist_100_82.pt'

    # Tensorboard
    writer = SummaryWriter('runs/evaluate_model_type_' + str(model_arch))

    # Get required dataset
    if test_data == 'mnist':
        # MNIST test data
        test_loader = torch.utils.data.DataLoader(DatasetHelper.getDataSet(False), batch_size=batch_size)
    elif test_data =='affnist':
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
        # Caps_Net_MNIST model
        loaded_network = MNISTCapsuleNetworkModel()
    elif model_arch == 2:
        # Caps_Net_ShiftedMNIST model
        loaded_network = ShiftedMNISTCapsuleNetworkModel()
    elif model_arch == 3:
        # Caps_Net_MultiMNIST model
        loaded_network = MultiMNISTCapsuleNetworkModel()
    elif model_arch == 4:
        # CNN_ShiftedMNIST model
        loaded_network = ShiftedMnistCNN()
    elif model_arch == 5:
        # CNN_MNIST model
        loaded_network = MnistCNN()
    elif model_arch == 6:
        # CNN_MultiMNIST model
        loaded_network = MultiMnistCNN()

    helper = Helper()
    table, total_params = helper.count_parameters(loaded_network)
    print(table)
    print('Total trainable parameters: ', total_params)

    loaded_network.load_state_dict(model_dict)
    evaluate(loaded_network, test_loader, batch_size, model_arch, writer, test_data, isResized)

    writer.flush()
    writer.close()
