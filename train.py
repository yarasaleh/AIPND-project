# PROGRAMMER: Yara Saleh
# Date: 1 July 2019

from train_functions import *
import argparse
import os 
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets, models, transforms


def main():
    args = get_input_args()
    input_size = None
    output_size = None
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_dir = os.path.join(args.save_dir , 'checkpoint.pth')
    
    
    if args.model == 'vgg16':
        input_size = 25088
        output_size = 102
        #loading the pre-trained model VGG
        model = models.vgg16(pretrained=True)
        # freeze the parameters so we dont backprop them (update the weights)
        for param in model.parameters():
            param.requires_grad = False    
        # untrained feed-forward network
        model.classifier = nn.Sequential(nn.Linear(25088,4096),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(4096,102),
                                         nn.LogSoftmax(dim=1))
        
    elif args.model == 'densenet121':
        input_size = 1024
        output_size = 102
        #loading the pre-trained model densenet121
        model = models.densenet121(pretrained=True)
        # freeze the parameters so we dont backprop them (update the weights)
        for param in model.parameters():
            param.requires_grad = False
        # untrained feed-forward network
        model.classifier = nn.Sequential(nn.Linear(1024,550),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(550,200),
                                        nn.ReLU(),
                                        nn.Linear(200,102),
                                        nn.LogSoftmax(dim=1))
        
 
    data_parser(args.data_path)
    
    if args.cuda :
        model = model.cuda()
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    train_model(model , data_parser.trainloader , data_parser.validloader ,criterion = criterion , optimizer = optimizer , epochs = int(args.epochs) , cuda = args.cuda)
    check_test_accuracy(data_parser.testloader,model,cuda = args.cuda)
    classifier = model.classifier
    image_dataset = data_parser.train_data
    model.class_to_idx = image_dataset.class_to_idx
    checkpoint = {'Structure': model,
             'input_size': input_size,
             'output_size':output_size,
             'epochs': args.epochs,
             'class_to_idx': model.class_to_idx,
             'classifier': classifier,
             'optimizer': optimizer.state_dict(),
             'state_dict': model.state_dict()}
    
    torch.save(checkpoint, 'checkpoint.pth')

main()
