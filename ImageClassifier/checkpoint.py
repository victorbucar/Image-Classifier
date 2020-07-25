import torch
from torch import optim
from torchvision import models


def save_checkpoint(model, optimizer, args, train_data):
    # Define checkpoint with parameters to be saved
    checkpoint = {
                 'arch':args.arch,
                 'epochs': args.epochs,
                 'classifier' : model.classifier,
                 'learning_rate': args.learning_rate,
                 'class_to_idx': train_data.class_to_idx,
                 'optimizer': optimizer.state_dict,
                 'state_dict': model.state_dict()
                 }
    #Save checkpoint
    torch.save(checkpoint, args.save_dir+'checkpoint.pth')
    
def choose_arch(func):
    dispatcher = { 'models.vgg11' : models.vgg11, 'models.alexnet' : models.alexnet, 'models.densenet121': models.densenet121}
    try:
        return dispatcher[func](pretrained=True)
    except:
        return "Invalid architecture, only 3 archs available vgg11/alexnet/densenet121"
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage) # this 'map_location=lambda storage, loc: storage' will let your model to load on cpu even if trained on gpu
    arch = 'models.'+checkpoint['arch']
    
    model = choose_arch(arch)
    
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['learning_rate'])
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model, optimizer
    