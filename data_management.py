from torchvision import datasets, transforms, models
import torch
from PIL import Image
def load_data(path):
    print("Loading and preprocessing data from {} ...".format(path))
    valid_dir = path + '/valid'
    train_dir = path + '/train'
    test_dir = path + '/test'
    
   
    test_transform = transforms.Compose([transforms.Resize(255),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])])
    valid_transform = transforms.Compose([transforms.Resize(255),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])

    train_transform = transforms.Compose([transforms.RandomRotation(50),
                                                  transforms.RandomResizedCrop(224),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                                       [0.229, 0.224, 0.225])])

    

   
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transform)
    train_data = datasets.ImageFolder(train_dir, transform = train_transform)
    test_data = datasets.ImageFolder(test_dir, transform = test_transform)

    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)
    
       
    return train_data, trainloader, validloader, testloader

def process_image(image):
    
    image = Image.open(image)
    
    image_transform = transforms.Compose([transforms.Resize(255),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])])
    
    return image_transform(image)