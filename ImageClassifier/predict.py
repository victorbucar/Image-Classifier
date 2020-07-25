#comment
import torch
import numpy as np
import json
from get_input_args import get_prediction_args
from checkpoint import load_checkpoint
from PIL import Image


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
# TODO: Process a PIL image for use in a PyTorch model

    im = Image.open(image)
    width, height = im.size
    im.thumbnail((256,height/width*256))
    new_width, new_height = im.size # take the size of resized image
    left = (new_width - 224)/2
    top = (new_height - 224)/2
    right = (new_width + 224)/2
    bottom = (new_height+ 224)/2
    im=im.crop((left, top, right, bottom))
    
    np_image=np.array(im)
    np_image = np_image / 255
    means=np.array([0.485, 0.456, 0.406])
    std= np.array([0.229, 0.224, 0.225])
    np_image=(np_image-means)/std
    np_image = np_image.transpose((2,0,1))
    #return torch.FloatTensor([np_image]) #pass a tensor 
    return np_image

def predict(args, model):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu == True  else "cpu")
    #image = process_image(image_path)
    #image = image.to(device)
    #image = image.unsqueeze_(0)
    
    model.eval()
    model = model.to(device)
    img = process_image(args.files[0])
    #image = torch.tensor(img.type(torch.FloatTensor))
    
    # We need a tensor for the model so change the image to a np.Array and then a tensor
    image = torch.from_numpy(np.array([img])).float()
    with torch.no_grad():
        #output = model(image)
        output = model(image.to(device))
        ps = torch.exp(output)
    
        top_p, top_class = ps.topk(args.top_k, dim=1)

    
        idx_to_class = {value: key for key,value in model.class_to_idx.items()}
        prob = [p.item() for p in top_p[0].data]
        
        classes = [idx_to_class[i.item()] for i in top_class[0].data]
        #print(idx_to_class)
        #print(classes)
        model.train()
        
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        
        # Get the lables from the json file
        labels = []
        for c in classes:
            labels.append(cat_to_name[c])
        #print(labels)
    return prob, labels

def main():
    
    in_args = get_prediction_args()
    
    chk_suffix = '.pth'
    checkpoint_path = in_args.files[1]+chk_suffix

    model, optimizer = load_checkpoint(checkpoint_path)
    
    prob, labels = predict(in_args, model)
    print('Probabilities of each class: ', prob)
    print('Classes predicted: ', labels)

    
main()