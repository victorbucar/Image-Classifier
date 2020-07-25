import torch
import torch.nn.functional as F
from torchvision import models
from torch import nn
from torch import optim
import checkpoint

def choose_arch(func):
    dispatcher = { 'models.vgg11' : models.vgg11, 'models.alexnet' : models.alexnet, 'models.densenet121': models.densenet121}
    try:
        return dispatcher[func](pretrained=True)
    except:
        return "Invalid architecture, only 3 archs available vgg11/alexnet/densenet121"
  
    
def build_classifier(archi, hidden_units, learn_rate, gpu=False):
    #check if gpu is avaliable
    device = torch.device("cuda" if torch.cuda.is_available() and gpu==True else "cpu")
    print('Building classifier... will use {} as device: '.format(device))
    #get our architecture model
    arch_function = 'models.' + archi
    #call the right architecture function to build our model
    model = choose_arch(arch_function)
    # map the entry layer units to the available models
    initial_units_map = { 'vgg11' : 25088, 'alexnet' : 9216, 'densenet121': 1024}
    initial_units = initial_units_map[archi]
    
    # Freeze parameters so we don't backprop through them, once they are pretrained
    # Only train the classifier parameters, feature parameters are frozen
    for param in model.parameters():
        param.requires_grad = False
        
    # define your model classifier    
    model.classifier = nn.Sequential(nn.Linear(initial_units, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units, 500),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(500,102),
                                     nn.LogSoftmax(dim=1))
    
    #define the loss function
    criterion = nn.NLLLoss()
    #assign the model to the right device cpu or gpu
    model.to(device)
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    return model, optimizer, criterion, device
    
def train_network(trainloader, validloader, args, train_data=None):
    #build classifier with the specific arguments
    model, optimizer, criterion, device = build_classifier(args.arch, args.hidden_units, args.learning_rate, args.gpu)
    running_loss, steps = 0,0
    print_every = 40
    model.train()
    print('Training the network')
    for epoch in range(args.epochs):
        for inputs, labels in trainloader:
            steps += 1
            
            #move to the right device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad() #zero the optimizer gradient value
            
            log_pred = model(inputs) #calculate model prediction
            
            loss = criterion(log_pred, labels) #calculate model loss
            
            loss.backward() #backpropagte the loss
            
            optimizer.step() #update the weights and bias
            
            running_loss += loss.item() # sum the loss
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval() # enable model evaluation mode for faster calculations
                with torch.no_grad(): # also no need to propagate gradients, reduce memory consumption for computations
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        v_pred = model.forward(inputs)
                        batch_loss = criterion(v_pred, labels)
                        valid_loss += batch_loss.item()
                        
                        #calculate model accuracy
                        ps = torch.exp(v_pred)
                        top_probs, top_labels = ps.topk(1, dim=1)
                        equals = top_labels == labels.view(*top_labels.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print("Epoch: {}/{}.. ".format(epoch+1, args.epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}..".format(valid_loss/len(validloader)),
                      "Test accuracy: {:.3f}..".format(accuracy/len(validloader) * 100)
                     )
                     
            
                running_loss = 0
                
                model.train()
                
    print("Finished the training and test!")
    if train_data != None:
        checkpoint.save_checkpoint(model, optimizer, args, train_data)
        print("Checkpoint saved!")
    
def test_network(testloader, validloader, args, epochs):
    train_network(testloader, validloader, in_args)
    



