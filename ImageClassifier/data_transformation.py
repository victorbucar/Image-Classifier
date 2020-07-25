#the purpose of this class is to have a file that handles our data transformations
#for training while training the model
import torch
from torchvision import datasets, transforms, models

def transform_data_sets(train_dir, test_dir, valid_dir, batch_size=64):
    """
    This function receive 3 directories and perform the necessary transformations 
    like rotate, crop, normalize and parse into a torch tensor, also determine the batch size 
    for each set.
    1. train_dir string, the path to the training images
    2. test_dir string, the path to the testing images
    3. valid_dir string, the path to the validation images
    4. batch_size integer, the size of the batch for each data set, default is 64.
    """
    # use the transforms from torch to transform our images for training
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    
    # use the same transforms a bit different for our testing and validation
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])])
    
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])])
    
    # load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    return train_data, test_data, valid_data

def create_data_loaders(train_dir, test_dir, valid_dir, batch_size=64):
    
    train_data, test_data, valid_data = transform_data_sets(train_dir, test_dir, valid_dir)
    # define the data loaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    
    return trainloader, testloader, validloader
