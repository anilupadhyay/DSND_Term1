# Basic usage: python predict.py /path/to/image checkpoint
# Options:
# Return top KK most likely classes: python predict.py input checkpoint --top_k 3
# Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
# Use GPU for inference: python predict.py input checkpoint --gpu

import json
import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description='Predict image class with checkpoint model')
parser.add_argument('input', action='store', help='image path')
parser.add_argument('checkpoint', action='store', 
                    type=str, help='checkpoint file path')
parser.add_argument('--top_k', action='store', default=5, 
                    type=int, help='top k classes')
parser.add_argument('--category_names', action='store', default='cat_to_name.json', 
                    type=str, help='category to name mapping json file')
parser.add_argument('--gpu', action='store_true', help='activate gpu', default=True)

result = parser.parse_args()

with open(result.category_names, 'r') as f:
    cat_to_name = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() and result.gpu else "cpu")
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    if checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif checkpoint['arch'] == 'vgg11':
        model = models.vgg11(pretrained=True)
    else:
    #     default to densenet121
        model = models.densenet121(pretrained=True)
        
    model = models.densenet121(pretrained=True)

    model.classifier = nn.Sequential(nn.Linear(checkpoint['input_size'], checkpoint['hidden_size']),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(checkpoint['hidden_size'], checkpoint['output_size']),
                                     nn.LogSoftmax(dim=1))

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    model.to(device)

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    # Open Image
    im = Image.open(image)

    # thumbnail
    width, height = im.size
    shortest = min(width, height)
    width = int(256 * width / shortest)
    height = int(256 * height / shortest)
    im.thumbnail((width, height))

    # crop
    left = (width - 224) / 2
    upper = (height - 224) / 2
    right = (width + 224) / 2
    lower = (height + 224) / 2
    im = im.crop((left, upper, right, lower))

    # numpy array
    np_image = np.array(im) / 255

    # normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # transpose
    np_image = np_image.transpose((2,0,1))
    return np_image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    
    img = torch.tensor(image).float()
    img = img.unsqueeze(0)
    
    idx_to_class = dict(map(reversed, model.class_to_idx.items()))
    
    with torch.no_grad():
        model.eval()
        
        img = img.to(device)
        log_ps = model.forward(img)
        ps = torch.exp(log_ps)
        top_p, top_idx = ps.topk(topk, dim=1)

        top_p = top_p.cpu().numpy().squeeze().tolist()
        top_idx = top_idx.cpu().numpy().squeeze().tolist()

        top_class = [idx_to_class[each] for each in top_idx]        
        
    return top_p, top_class

model = load_checkpoint(result.checkpoint)
probabilities, classes = predict(result.input, model, result.top_k)
print(probabilities, [cat_to_name[each] for each in classes])    
