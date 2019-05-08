# Train a new network on a data set with train.py

# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
# Options:
# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
# Choose architecture: python train.py data_dir --arch "vgg13"
# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# Use GPU for training: python train.py data_dir --gpu

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse

parser = argparse.ArgumentParser(description='Train a new network on a Image data set')
parser.add_argument('data_dir', action='store', help='training image directory')
parser.add_argument('--save_dir', action='store', default='./',
                    type=str, help='checkpoint directory')
parser.add_argument('--arch', action='store', default='densenet121',
                    type=str, help='transfer learning model architecture ')
parser.add_argument('--learning_rate', action='store', default=0.003, 
                    type=float, help='model learning rate')
parser.add_argument('--hidden_units', action='store', default=256, 
                    type=int, help='hidden unit count')
parser.add_argument('--epochs', action='store', default=3, 
                    type=int, help='epoch count')
parser.add_argument('--gpu', action='store_true', help='activate gpu', default=True)

result = parser.parse_args()

data_dir = result.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform = test_transforms)
test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(test_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() and result.gpu else "cpu")

if result.arch == 'vgg13':
    model = models.vgg13(pretrained=True)
elif result.arch == 'vgg11':
    model = models.vgg11(pretrained=True)
elif result.arch == 'densenet121':
    model = models.densenet121(pretrained=True)
elif result.arch == 'resnet101':
    model = models.resnet101(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
model.classifier = nn.Sequential(nn.Linear(1024, result.hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(result.hidden_units, 102),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=result.learning_rate)

model.to(device);

epochs = result.epochs
steps = 0
running_loss = 0
print_every = 25
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()

            model.class_to_idx = train_data.class_to_idx

checkpoint = {'input_size': 1024,
              'output_size': 102,
              'hidden_size': result.hidden_units,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx,
              'arch': result.arch}

torch.save(checkpoint, result.save_dir + 'checkpoint.pth')
       