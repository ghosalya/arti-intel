'''
Homework 4 - Upsampling
Coding part 2
'''
import numpy as np
import os

def from_text(filename):
    data = []
    label = []
    with open(filename, 'r') as infile:
        for line in infile.readlines():
            x, y, lb = line.split()
            data.append([float(x), float(y)])
            label.append([float(lb)])
    data = np.asarray(data)
    label = np.asarray(label)
    return data, label

import torch
from torch.utils.data import Dataset, DataLoader
import PIL.Image as Image

# import torchvision
# from torchvision import models, transforms, utils

class ImbaDataset(Dataset):
    def __init__(self, root_path, mode='train'):
        self.meta = {'root_path': root_path,
                     'mode': mode}
        assert mode in ['train', 'test']
        if mode == 'train':
            data, label = from_text(os.path.join(root_path, 'samplestr.txt'))
        elif mode == 'test':
            data, label = from_text(os.path.join(root_path, 'sampleste.txt'))
        
        self.dataset = data 
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return {'data': torch.from_numpy(self.dataset[index]),
                'label': torch.from_numpy(self.label[index])}

