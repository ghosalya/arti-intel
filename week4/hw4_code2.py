'''
Homework 4 - Upsampling
Coding part 2
'''
import numpy as np
import os
import matplotlib.pyplot as plt

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

def plot_dataset(data, label):
    true_mask = label.reshape((-1)).astype(bool)
    false_mask = (1-label).reshape((-1)).astype(bool)

    data_true = np.transpose(data[true_mask])
    data_false = np.transpose(data[false_mask])

    print('samples with positive class:', len(data_true[0]),
          '\nsamples with negative class:', len(data_false[0]))

    plt.figure()
    plt.plot(data_true[0], data_true[1], '.b')
    plt.plot(data_false[0], data_false[1], '.r')

def draw_model(model):
    w = model.state_dict()['fc.weight']
    b = model.state_dict()['fc.bias']
    w1 = w[0,0].item()
    w2 = w[0,1].item()
    b1 = b[0].item()

    m = -w1/w2
    c = -b1/w2

    line_x = [ i/100 for i in range(-300, 700)]
    line_y = [ m*i + c for i in line_x]
    
    shade_x = [line_x[0], line_x[-1]]
    shade_y = [line_y[0], line_y[-1]]

    if w2 > 0:
        shade_y += [max(shade_y)]
        if m > 0:
            shade_x += [min(shade_x)]
        else:
            shade_x += [max(shade_x)]
    else:
        shade_y += [min(shade_y)]
        if m > 0:
            shade_x += [max(shade_x)]
        else:
            shade_x += [min(shade_x)]

    # plot_dataset(data_tr, label_tr)
    plt.plot(line_x, line_y, '.m', lw=1)
#     plt.line()
    plt.fill(shade_x, shade_y, '.c')