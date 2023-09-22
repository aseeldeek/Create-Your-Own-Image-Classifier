"""
Author:     Aseel Deek 
Date:       16 September 2023
Project:    Image Classifier Project
CopyRight:  Udacity.com 
How to run the script: 
        python train.py "flowers" --save_dir "test_checkpoint.pth" --arch "vgg16" --learning_rate 0.002 --hidden_units 4096  --epochs 6 --gpu
    OR: 
        python train.py "flowers" --arch "vgg16" --gpu

"""

# all the imports needed for the script
import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torchvision import models
import argparse
import json
import time



def parse_command_line_args():
    # define the arguments and their values in a dictionary
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_dir', metavar='data_dir', type=str)
    # default path to save the model
    parser.add_argument('--save_dir', action='store', dest='save_dir', type=str, default='checkpoint.pth')
    
    # default architecture to use if not specified
    parser.add_argument('--arch', action='store', dest='arch', type=str, default='vgg16', choices=['vgg16', 'densenet121'])
    
    # default learning rate for Adam optimizer
    parser.add_argument('--learning_rate', action='store', dest='learning_rate', type=float, default=0.001)
    
    # default number of hidden units
    parser.add_argument('--hidden_units', action='store', dest='hidden_units', type=int, default=4096)
    
    # default number of epochs to train the model
    parser.add_argument('--epochs', action='store', dest='epochs', type=int, default=7)
    
    # default is to use CPU
    parser.add_argument('--gpu', action='store_true', default=False)
    return parser.parse_args()



def loading_model(arch, hidden_units, gpu):
    if gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        num_in_features = 25088
    else:
        model = models.densenet121(pretrained=True)
        num_in_features = 1024

   # Freeze the pre-trained model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define feed-forward classifier
    classifier = nn.Sequential(
        nn.Linear(num_in_features, hidden_units),  # number of hidden units as needed
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, 102),  # 102 is the number of flower categories
        nn.LogSoftmax(dim=1)
    )

    # Replace the classifier in the VGG model
    model.classifier = classifier
    model.to(device)
    return model, device, num_in_features


def training_model(epochs, trainloader, validateloader, model, device, criterion, optimizer):
    start = time.time()
    print("...................................................\n")
    print('Training The Module Has Started...')
    print("...................................................\n")
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()  # Set the model to training mode
        for inputs, labels in trainloader:
            
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate training loss
        avg_train_loss = running_loss / len(trainloader)

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for inputs, labels in validateloader:
                # inputs: are images, labels: are the corresponding labels
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        # Calculate validation loss and accuracy
        avg_val_loss = val_loss / len(validateloader)
        accuracy = (correct_predictions / total_predictions) * 100

        # Print training and validation metrics
        print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {avg_train_loss:.3f} - Validation Loss: {avg_val_loss:.3f} - Validation Accuracy: {accuracy:.2f}%")

    end = time.time()
    total_time = end - start
    print(">>> Training Completed Successfully...")
    print("...................................................\n")
    print("The Training Duration is :{:.0f} min {:.0f} sec".format(total_time // 60, total_time % 60))



def save_checkpoint(model, image_datasets, optimizer, epochs, hidden_units, arch, input_size, output_size, learning_rate, file_path):
    checkpoint = {
    'model_state_dict': model.state_dict(),
    'class_to_idx': image_datasets['train'].class_to_idx,
    'optimizer_state_dict': optimizer.state_dict(),
    'classifier': model.classifier,
    'epochs': epochs,
    'hidden_units': hidden_units,
    'pretrained_model': arch,
    'input_size': input_size,
    'output_size': output_size,
    'learning_rate': learning_rate
}

    torch.save(checkpoint, file_path)
    print("Model saved to storage...")

def main():
    
    # getting the arguments from the command line
    args = parse_command_line_args()
    
    data_dir = args.data_dir # 'flowers'
    train_dir = data_dir + '/train' # 'flowers/train'
    valid_dir = data_dir + '/valid' # 'flowers/valid'
    test_dir = data_dir + '/test'   # 'flowers/test'

    # Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # mean and std for the pre-trained model
    ]),
        'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # mean and std for the pre-trained model
    ]),
        'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # mean and std for the pre-trained model
    ])
    }
    
    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']), # data_transforms['train'] is a composition of transforms
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']), # data_transforms['valid'] is a composition of transforms 
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']) # data_transforms['test'] is a composition of transforms
    }
    
    # Using the image datasets and the transforms, define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    }
    
    # loading the json file into a dictionary
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f) # loading the json file into a dictionary

    # getting model, device object and number of input features
    model, device, num_in_features = loading_model(args.arch, args.hidden_units, args.gpu)
    # print(model)
    
    criterion = nn.NLLLoss() # Negative Log Likelihood Loss
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate) # Adam optimizer

    # training the model
    training_model(args.epochs, dataloaders['train'], dataloaders['valid'], model, device, criterion, optimizer) 

    # 'checkpoint.pth' is the default path to save the model
    file_path = args.save_dir 

    # number of flower categories
    output_size = 102 
    
    # saving the model
    save_checkpoint(model, image_datasets, optimizer,args.epochs, args.hidden_units, 
                    args.arch, num_in_features, output_size, args.learning_rate, file_path) # saving the model


if __name__ == "__main__":
    main()
    
# import torch
# import torchvision
# import torchaudio
# print("Torch version:", torch.__version__)
# print("Torchvision version:", torchvision.__version__)
# print("Torchaudio version:", torchaudio.__version__)