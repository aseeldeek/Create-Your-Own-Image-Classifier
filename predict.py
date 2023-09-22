"""
Author:     Aseel Deek 
Date:       16 September 2023
Project:    Image Classifier Project
CopyRight:  Udacity.com
Last modified:  22 September 2023
How to run the script:
        python predict.py flowers/test/60/image_02978.jpg checkpoint.pth --top_k 3 --category_names cat_to_name.json --gpu
    OR: 
        python predict.py flowers/test/60/image_02978.jpg checkpoint.pth --gpu

"""
import argparse
import json
import numpy as np
from PIL import Image
import torch
import torchvision
import torch.nn.functional as F





def parse_command_line_args():
    """
        this function is used to parse the arguments passed to the script
        :return: the parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', metavar='image_path', type=str, default='flowers/test/58/image_02663.jpg')
    parser.add_argument('checkpoint', metavar='checkpoint', type=str, default='checkpoint.pth')
    parser.add_argument('--top_k', action='store', dest="top_k", type=int, default=5)
    parser.add_argument('--category_names', action='store', dest='category_names', type=str, default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', default=False)
    return parser.parse_args()


def load_checkpoint(filepath):
    """
        this function is used to load the checkpoint file
        :param filepath: the path to the checkpoint file, default is checkpoint.pth, type is string
        :return: the model
    """
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    my_model = getattr(torchvision.models, checkpoint['pretrained_model'])(pretrained=True)
    my_model.learning_rate = checkpoint['learning_rate']
    my_model.hidden_units = checkpoint['hidden_units']
    my_model.classifier = checkpoint['classifier']
    my_model.load_state_dict(checkpoint['model_state_dict'])
    my_model.class_to_idx = checkpoint['class_to_idx']
    my_model.optimizer = checkpoint['optimizer_state_dict']
    my_model.input_size = checkpoint['input_size']
    my_model.output_size = checkpoint['output_size']
    my_model.epochs = checkpoint['epochs']
    return my_model


def process_image(image):
    """
        this function is used to process the image
        { Scales, crops, and normalizes a PIL image for a PyTorch model }
        input: image path
        output: tensor of the image
    """
    # Load the image using PIL
    img = Image.open(image)
    # reize
    img.resize((256,256))
    
    # centre crop
    width, height = img.size   # Get dimensions
    new_width, new_height = 224, 224
    
    left = round((width - new_width)/2)
    top = round((height - new_height)/2)
    x_right = round(width - new_width) - left
    x_bottom = round(height - new_height) - top
    right = width - x_right
    bottom = height - x_bottom

    # crop the center of the image
    img = img.crop((left, top, right, bottom))
    
    # convert colour channel from 0-255, to 0-1
    np_image = np.array(img)/255
    
    # normalize for model
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    
    # tranpose color channge to 1st dim
    np_image = np_image.transpose((2 , 0, 1))
    
    # convert to Float Tensor
    tensor = torch.from_numpy(np_image)
    tensor = tensor.type(torch.cuda.FloatTensor)
    
    # return tensor
    return tensor



def predict_img(image_path, model, top_k, gpu):
    """
        this function is used to predict the image class
        using a trained model
        input: image path, model, top_k: top K classes, gpu
        output: top_p: probability of top K predicted clasees, 
                top_classes: the top K predicted classes
    """
    # TODO: edit this function so your code in image classifier.ipynb will work
    if gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model.to(device)

    image = Image.open(image_path)
    image = process_image(image_path)
    image = torch.tensor(image).unsqueeze(0).float().to(device)
    model.eval() 
    
    with torch.no_grad():
        image = image.to(device)  # Move the image tensor to the GPU
        output = model.forward(image) # Forward pass

    probabilities = F.softmax(output.data, dim=1) # Calculate the class probabilities (softmax) for img

    topk_probabilities = np.array(probabilities.topk(top_k)[0][0].cpu().numpy())  # Convert from Tensor to Numpy array

    idx_to_class = {v: k for k, v in model.class_to_idx.items()} # Invert the class_to_idx dictionary
    top_classes = [int(idx_to_class[each]) for each in np.array(probabilities.topk(top_k)[1][0].cpu())] # Get the indices of the top 5 classes



    return topk_probabilities, top_classes


def load_categort_names(category_names_file):
    with open(category_names_file) as file:
        category_names = json.load(file)
    return category_names


def main():
    
    # get the arguments
    args = parse_command_line_args()
    
    # save the image path  from the user
    image_path = args.image_path
    # print(image_path)
    
    # save the entered checkpoint file from the user
    checkpoint = args.checkpoint
    
    # save the top_k value from the user
    top_k = args.top_k
    
    # save the category_names file from the user
    category_names = args.category_names
    
    # save the gpu value from the user
    gpu = args.gpu

    # load the model from the checkpoint file
    model = load_checkpoint(checkpoint)

    # predict the image
    prob, topk_classes = predict_img(image_path, model, top_k, gpu)

    # load the category names
    category_names = load_categort_names(category_names)

    # get the labels
    labels = [category_names[str(index)] for index in topk_classes]

    # print the results
    print(f" >>>>>>>>> Results for your File: {image_path}")
    print(labels)
    print(prob)
    print("\n")

    # print the results in a more readable way
    for i, (label, probability) in enumerate(zip(labels, prob), start=1):
        print(f"{i} - {label} with a probability of {probability * 100:.2f}%")



if __name__ == "__main__":
    main()
