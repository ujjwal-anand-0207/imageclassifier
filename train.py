import os
import json
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from collections import OrderedDict
from torch import nn,optim
from torchvision import datasets,models,transforms

def validation():
    print("VALIDATION PARAMETERS")
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled bur no GPU detected")
    if (not os.path.isdir(args.data_directory)):
        raise Exception("Directory do not exists")
    data_dir=os.listdir(args.data_directory)
    if (not set(data_dir).issubset({'test','train','valid'})):
        raise Exception("missing sub directories")
    if args.arch not in ('vgg','densenet',None):
        raise Exeception("please choose vgg or densenet")
    
def process_data(data_dir):
    print("Processing of data into iterators")
    train_dir,test_dir,valid_dir=data_dir
    
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485,0.456,0.406),(0.229, 0.224, 0.225)),
                                 ])
    data_transforms = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                 ])

    test_datasets = datasets.ImageFolder(test_dir,transform=transform)
    train_datasets=datasets.ImageFolder(train_dir,transform=transform)
    valid_datasets=datasets.ImageFolder(valid_dir,transform=transform)
    
    testloader = torch.utils.data.DataLoader(test_datasets,batch_size=32,shuffle=True)

    validloader = torch.utils.data.DataLoader(valid_datasets,batch_size=32,shuffle=True)

    trainloader = torch.utils.data.DataLoader(train_datasets,batch_size=32,suffle=True)
    
    with open('cat_to_name.json','r')as f:
        cat_to_name =json.load(f)
    loaders = {'train':trainloaders,'valid':validloaders,'test':testloaders,'labels':cat_to_name}
    return loaders

def get_data():
    valid_dir = args.data_directory + '/valid'
    test_dir =args.data_directory + '/test'
    train_dir=args.data_directory+ '/train'
    data_dir = [train_dir,test_dir,valid_dir]
    return process_data(data_dir)

def build_model(data):
    if (args.arch is None):
        arch_type='vgg'
    else:
        arch_type=args.arch
    if (arch_type=='vgg'):
        model = models.vgg19(pretrained=True)
        input_node=25088
    elif (arch_type =='densenet'):
        model =models.densenet121(pretrained=True)
        input_node=1024
    if (args.hidden_units is None):
        hidden_units =4096
    else:
        hidden_units= args.hidden_units
    for param in model.parameters():
        param.requires_grad = False
        
    hidden_units= int(hidden_units)
    
    classifier = nn.Sequential(OrderedDict([('l1',nn.Linear(input_node,hidden_units)),('relu',nn.ReLU()),       ('l2',nn.Linear(hidden_units,102)),    ('output',nn.LogSoftmax(dim=1))
                                ]))
        
    model.classifier =classifier
    return model
        
def test_accuracy(model,loader,device='cpu'):
    correct=0
    total=0
    with torch.no_grad():
         for data in loader:
             images,labels =data[0].to(device),data[1].to(device)
             output=model(images)
             _, predicted =torch.max(output.data,1)
             total += labels.size(0)
             correct +=(predicted==labels).sum().item()
    return correct/total    
        
def train(model,data):
        print_every=50
        if (args.learing_rate is None):
           learn_rate =0.001
        else:
           learn_rate =args.Learning_rate
        if (args.epochs is None):
           epochs=3
        else:
           epochs=args.epochs
        if (args.gpu):
           device='cuda'
        else:
           device ='cpu'
        
        
        
        
        learn_rate=float(learn_rate)
        epochs=int(epochs)
        
        testloader=data['test']
        trainloader=data['train']
        validloader=data['valid']
        steps=0
        criterion =nn.NLLLoss()
        optimizer=optim.Adam(model.clasifier.parameters(),lr=learn_rate)
        model.to(device)
        
        for e in range(epochs):
            running_loss=0
            for li,(inputs,labels) in enumerate(trainloader):
                step+=1
                inputs,labels=inputs.to(device),labels.to(device)
                optimizer.zero_grad()
                output=model.forward()
                loss=criterion(output,labels)
                optimizer.step()
        
                running_loss += loss.item()
                if steps % print_every ==0:
                   valid_accuracy= test_accurcy(model,validloader,device)
                   print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Accuracy: {}".format(round(valid_accuracy,4)))            
                   running_loss = 0
        test_result = test_accuracy(model,testloader,device)
        print('final accuracy:{}'.format(test_result))
        return model

def save_model(model):
    print("saving model")
    if (args.save_dir is None):
        save_dir = 'check.pth'
    else:
        save_dir = args.save_dir
    checkpoint = {
                'model': model.cpu(),
                'features': model.features,
                'classifier': model.classifier,
                'state_dict': model.state_dict()}
    torch.save(checkpoint, save_dir)
    return 0
      
def create_model():
    validation()
    data = get_data()
    model = build_model(data)
    model = train(model,data)
    save_model(model)
    return None

def parse():
    parser = argparse.ArgumentParser(description='Train a neural network with open of many options!')
    parser.add_argument('data_directory', help='data directory (required)')
    parser.add_argument('--save_dir', help='directory to save a neural network.')
    parser.add_argument('--arch', help='models to use OPTIONS[vgg,densenet]')
    parser.add_argument('--learning_rate', help='learning rate')
    parser.add_argument('--hidden_units', help='number of hidden units')
    parser.add_argument('--epochs', help='epochs')
    parser.add_argument('--gpu',action='store_true', help='gpu')
    args = parser.parse_args()
    return args

def main():
    print("creating a deep learning model")
    global args
    args = parse()
    create_model()
    print("model finished!")
    return None

main()
        

        
        
        
   
       
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        