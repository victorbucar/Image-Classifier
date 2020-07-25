import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from tqdm import tqdm_notebook as tqdm
import time
from PIL import Image
import numpy as np

from get_input_args import get_input_args
import data_transformation
import create_network
def main():
    
    #get the arguments from the user
    in_args = get_input_args()
    
    # data directories
    data_dir = in_args.data_dir
    
    train_dir = data_dir + 'train'
    valid_dir = data_dir + 'valid'
    test_dir = data_dir + 'test'
    
    #use data_transformation to load the data with the appropriate transformations
    trainloader, testloader, validloader = data_transformation.create_data_loaders(train_dir, test_dir, valid_dir)
    
    train_data, test_data, valid_data = data_transformation.transform_data_sets(train_dir, test_dir, valid_dir)
    
    #model, optimizer, criterion = create_network.build_classifier(in_args.arch, in_args.hidden_units, in_args.learning_rate)
    create_network.train_network(trainloader, validloader, in_args, train_data)
    
    #test the network on test data
    epochs = 3
    
    #create_network.test_network(testloader, validloader, in_args, epochs)
    
# call for main function    
main()
