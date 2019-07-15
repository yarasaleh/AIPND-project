# PROGRAMMER: Yara Saleh
# Date: 4 July 2019

import json
import torch
import argparse
import numpy as np
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, models, transforms

def get_arguments():
    msg = 'predict.py takes two manditory arguments an image and checkpoint from the trained network'
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('input', action = 'store')
    parser.add_argument('checkpoint', action = 'store')
    #optional arguments
    parser.add_argument('--top_K', action = 'store',dest='top_K', default=5, help='Set the number of top results to view.')
    parser.add_argument("--category_names", action="store", dest="categories", default="cat_to_name.json",
                        help="Number of top results you want to view.")
    parser.add_argument('--gpu', action = 'store_true' , dest = 'cuda', default = True , help = 'Set Cuda True for using the GPU')

    return parser.parse_args()


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image_ratio = image.size[1] / image.size[0]
    image = image.resize((256, int(image_ratio*256)))
    half_the_width = image.size[0] / 2
    half_the_height = image.size[1] / 2
    image = image.crop((half_the_width - 112,
                       half_the_height - 112,
                       half_the_width + 112,
                       half_the_height + 112))
    
    image = np.array(image)
    image = image/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])
    
    image = (image - mean) / std_dev
    image = image.transpose((2, 0, 1))
    
    return torch.from_numpy(image)

def predict(image_path, model,cuda, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    if cuda:
        model.cuda()
    else:
        model.cpu()
    image = None
    model.eval()
    with Image.open(image_path) as img:
        image = process_image(img)
    with torch.no_grad():
        image = image.unsqueeze_(0)

    if cuda:
        image = image.cuda()
    output = model.forward(image.float())
    probability = F.softmax(output.data,dim=1)
    return probability.topk(topk)



def load_checkpoint(filepath,cuda):
    if cuda:
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)

    model = checkpoint['Structure']
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    epochs = checkpoint ['epochs']
    model.classifier = checkpoint['classifier']
    optimizer = checkpoint['optimizer']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model