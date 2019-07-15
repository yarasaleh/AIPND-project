# PROGRAMMER: Yara Saleh
# Date: 1 July 2019

import argparse
import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', action = 'store', dest = 'save_dir' , default = '.' , help = 'Set a directory to save checkpoints')
    parser.add_argument('--model', action = 'store' , dest = 'model' , default = 'vgg16' , help ='Set architechture densenet121 or vgg19')
    parser.add_argument('--epochs', action = 'store' , dest = 'epochs' , default = 7 , help = 'Set number of epochs')
    parser.add_argument("--learning_rate", action="store", dest="lr", default=0.003 , help = "Set learning rate")
    parser.add_argument('--hidden_units', action='store', dest='hidden_units', default=4096 , help = 'Set number of hidden units')
    parser.add_argument('--gpu', action = 'store_true' , dest = 'cuda', default = True , help = 'Set Cuda True for using the GPU')
    parser.add_argument('data_path', action = 'store')
    
    
    return parser.parse_args()

def data_parser(data_path):
    train_dir = data_path + '/train'
    valid_dir = data_path + '/valid'
    test_dir = data_path + '/test'
    
    data_parser.test_valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    data_parser.train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
        
    data_parser.train_data = datasets.ImageFolder(train_dir, transform=data_parser.train_transforms)
    data_parser.valid_data = datasets.ImageFolder(valid_dir, transform=data_parser.test_valid_transforms)
    data_parser.test_data = datasets.ImageFolder(test_dir, transform=data_parser.test_valid_transforms)
    
    data_parser.trainloader = torch.utils.data.DataLoader(data_parser.train_data,batch_size=64,shuffle=True)
    data_parser.testloader = torch.utils.data.DataLoader(data_parser.test_data,batch_size=64)
    data_parser.validloader = torch.utils.data.DataLoader(data_parser.valid_data,batch_size=64)
    

def train_model(model, trainloader,validloader, criterion, optimizer, epochs=7, cuda=True):
    epochs = 7
    steps = 0
    print_every = 5
    running_loss = 0
    if cuda and torch.cuda.is_available:
        model.cuda()
    else:
        model.cpu()
        
    for e in range(epochs):
        for inputs , labels in trainloader:
            steps+=1
            if cuda :
                inputs, labels = inputs.cuda(), labels.cuda()
                
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            
            if steps % print_every == 0 :
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs , labels in validloader:

                        if cuda :
                            inputs , labels = inputs.cuda() , labels.cuda()
                        output = model.forward(inputs)
                        batch_loss = criterion(output, labels)
                        valid_loss += batch_loss.item()

                        ps = torch.exp(output)
                        top_p , top_class = ps.topk(1,dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                "Accuracy: {:.3f}".format(accuracy/len(validloader)))
                running_loss = 0
                model.train()
                    
def check_test_accuracy(testloader,model,cuda = True):
    correct = 0
    total = 0
    model.eval()
    if cuda:
        model.cuda()
    else:
        model.cpu()
    with torch.no_grad():
        for data in testloader:
            images , labels = data
            if cuda :
                images , labels = images.cuda() , labels.cuda()
            output = model(images)
            _,predicted = torch.max(output.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("The accuracy of the network test: {}".format(100 * correct / total))
    
    
                    
                    

            

