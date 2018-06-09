'''
Homework 4 - FlowersStuff Dataset

`readflowerslabel.py` is hardcoded and doesnt return,
so this scrip will also include file parser
'''
import os

def get_dataset_paths(root_dir):
    '''
    Takes one root directory and return the train.txt, val.txt and jpg directory
    '''
    train_labels = os.path.join(root_dir, 'trainfile.txt')
    test_labels = os.path.join(root_dir, 'testfile.txt')
    val_labels = os.path.join(root_dir, 'valfile.txt')
    jpg_folder = os.path.join(root_dir, 'jpg')

    return train_labels, test_labels, val_labels, jpg_folder

'''
Dataset class for the current Flower training
'''
import torch
from torch.utils.data import Dataset, DataLoader
import PIL.Image as Image

import torchvision
from torchvision import models, transforms, utils

class FlowerDataset(Dataset):
    def __init__(self, root_path, transforms=None, mode='train'):
        '''
        Flower Dataset.
        '''
        self.meta = {'root_path': root_path, 
                     'transforms':transforms or self.get_default_transforms(), 
                     'mode':mode}
        assert mode in ['train','val','test']

        train, test, val, jpg = get_dataset_paths(root_path)
        self.jpg_folder = jpg

        if mode == 'train':
            self.dataset_path = train
        elif mode == 'val':
            self.dataset_path = val
        elif mode == 'test':
            self.dataset_path = test

        self.dataset = [] # list of "image.jpg label" as a space-separated string
        with open(self.dataset_path, 'r') as datafile:
            self.dataset = datafile.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        '''
        now we load the image and label
        '''
        imgname, label = self.dataset[index].split()
        image = Image.open(os.path.join(self.jpg_folder, imgname)).convert("RGB")
        image = self.meta['transforms'](image)
        return {'label':int(label), 'image':image}

    def get_default_transforms(self):
        '''
        In case no transform is passed
        '''
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


'''
Trainer
'''
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import math, time
import numpy as np

def train_model(dataset, model, optimizer, scheduler=None,
                batchsize=4, mode='train', epoch=5, use_gpu=True, print_every=100):
    '''
    run training
    '''
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=4)
    total_batch_count = (len(dataset) // batchsize) + 1
    criterion = torch.nn.CrossEntropyLoss()
    model.train(mode == 'train')

    for e in range(epoch):
        print('[{}] - Epoch {}..'.format(mode, e))
        epoch_start = time.clock()
        running_loss = 0.0
        running_corrects = 0 
        if scheduler: scheduler.step()

        iter = 0
        
        for data in dataloader:
            optimizer.zero_grad()
            inputs, labels = data['image'], data['label']
            iter += 1
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:    
                inputs, labels = Variable(inputs), Variable(labels)

            outputs = model.forward(inputs)
            _, predictions = outputs.max(dim=1)

            loss = criterion(outputs, labels) / len(dataset)
            if mode == 'train':
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            running_corrects += (predictions.cpu() == labels.cpu()).sum().item()/float(len(dataset))

            if (iter % print_every) == 0:
                print('                  ... iteration {}/{}'.format(iter, total_batch_count))

        epoch_time = time.clock() - epoch_start
        print("      >> Epoch loss {:.5f} accuracy {:.3f}        \
              in {:.4f}s".format(running_loss, running_corrects, epoch_time))

    return model

'''
Models
'''

def get_trained_resnet(numcl, use_gpu=True):
    model_ft = models.resnet18(pretrained=True)
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, numcl)
    if use_gpu:
        model_ft = model_ft.cuda(0)
    return model_ft

def get_empty_resnet(numcl, use_gpu=True):
    model_ft = models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, numcl)
    if use_gpu:
        model_ft = model_ft.cuda(0)
    return model_ft

def get_unfrozen_resnet(numcl, use_gpu=True):
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, numcl)
    if use_gpu:
        model_ft = model_ft.cuda(0)
    return model_ft

'''
Main Function
'''

def run_training(model_ft, use_gpu=True, limit=None):
    model_ft = get_trained_resnet(102)

    for learnrate in [1e-1, 1e-2, 1e-3]:
        print(" - with learnrate",learnrate)
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=learnrate, momentum=0.9)
        flower_dataset = FlowerDataset('..\\datasets\\flowersstuff\\102flowers\\flowers_data', mode='train')
        model_ft = train_model(flower_dataset, model_ft, optimizer_ft, 
                            epoch=1, mode='train', use_gpu=use_gpu, print_every=200)
        flower_dataset = FlowerDataset('..\\datasets\\flowersstuff\\102flowers\\flowers_data', mode='val')
        train_model(flower_dataset, model_ft, optimizer_ft, 
                    epoch=1, mode='val', use_gpu=use_gpu, print_every=1000)
    # return flower_dataset, model_ft

def main():
    print("Training ResNet Fully Loaded;")
    run_training(get_trained_resnet(102))

    print("Training ResNet Empty;")
    run_training(get_empty_resnet(102))

    print("Training ResNet with Unfrozen Last Layer")
    run_training(get_unfrozen_resnet(102))

if __name__ == '__main__':
    main()