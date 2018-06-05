import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import PIL.Image as Image

import torchvision
from torchvision import models, transforms, utils

import getimagenetclasses as ginc
import os, math, time
import numpy as np
import itertools


#####
#   Dataset subclass
#####

class Wk3Dataset(Dataset):
    def __init__(self, root_dir, file_prefix='ILSVRC2012_val_',
                 img_ext='.JPEG', val_ext='.xml', synset='synset_words.txt',
                 five_crop=False, data_limit=0, selector=None):
        '''
        NOTE: set up your root_dir directory to consist of 
        2 directories:
        - imagespart: where the images are
        - val: where the xml's are (for class values etc)
        '''
        self.meta = {'root_dir':root_dir,
                     'file_prefix':file_prefix,
                     'synset':synset,
                     'img_ext':img_ext, 'val_ext':val_ext,
                     'five_crop': five_crop}

        # assertion for the mentioned assumption
        assert os.path.exists(os.path.join(root_dir, 'imagespart'))
        assert os.path.exists(os.path.join(root_dir, 'val'))
        assert os.path.exists(os.path.join(root_dir, synset))

        # metadata
        self.classes = ginc.get_classes()
        _, s2i, s2d = ginc.parsesynsetwords(os.path.join(root_dir, synset))
        self.dataset = [filename[len(file_prefix):-len(img_ext)]
                        for filename in os.listdir(os.path.join(root_dir, 'imagespart'))]
        if data_limit > 0:
            self.dataset = self.dataset[:data_limit]

        if selector is not None:
            self.dataset = [d for d, s in zip(self.dataset, selector) if s]

        self._rev_dataset = s2i 
        self.data_description = s2d

    def get_val_path(self, index):
        return os.path.join(self.meta['root_dir'], 'val',
                            self.meta['file_prefix'] + str(index).zfill(8) + self.meta['val_ext'])

    def get_img_path(self, index):
        return os.path.join(self.meta['root_dir'], 'imagespart',
                            self.meta['file_prefix'] + str(index).zfill(8) + self.meta['img_ext'])
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        '''
        Only when __getitem__ is called should the
        code load the actual image
        '''
        # since filenames starts with 1, index should be incremented
        index = self.dataset[idx]
        # 1. get corresponding dataset metadata
        label, _ = ginc.parseclasslabel(self.get_val_path(index), self._rev_dataset)
        # index label
        label_vector = int(label)
        
        # 2. load the image file
        image = Image.open(self.get_img_path(index)).convert("RGB")
        image = self.transform_short(image)

        five_crop = self.meta['five_crop']
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        if five_crop:
            image = self.transform_fivecrop(image)
            image = [normalize(transforms.ToTensor()(t)) for t in image]
            image = torch.stack(image)
            # stack :- 5 [3, 224, 224] tensor into [5, 3, 224, 224]
            # cat:- 5 [3, 224, 224] tensor into [15, 224, 224]
        else:
            image = self.transform_centercrop(image)
            image = normalize(transforms.ToTensor()(image))

        itm = {'label':label_vector, 'image':image}
        return itm
        
    #   Transform functions - always returns a function
    def transform_short(self, image, short_size=280):
        '''
        Do the transformation:
        - resize till the shorter side is 280
        '''
        width, height = image.size
        ratio = short_size / min(width, height)
        new_size = (int(width*ratio), int(height*ratio)) 
        new_img = image.resize(new_size, Image.ANTIALIAS)
        return new_img

    def transform_centercrop(self, image, size=224):
        '''
        Do the transformation:
        - take the center crop of sizexsize
        '''
        width, height = image.size 
        left = (width // 2) - (size//2)
        upper = (height // 2) - (size//2)
        right = left + size
        lower = upper + size
        crop_image = image.crop((left, upper, right, lower))
        return crop_image

    def transform_fivecrop(self, image, size=224):
        width, height = image.size 
        center_crop = self.transform_centercrop(image, size=size)
        topleft_crop = image.crop((0, 0, size, size))
        topright_crop = image.crop((width - size, 0, width, size))
        botleft_crop = image.crop((0, height - size, size, height))
        botright_crop = image.crop((width - size, height - size, width, height))
        return [center_crop, topleft_crop, topright_crop, botleft_crop, botright_crop]


######
#   Training codes
######

def train_model(dataset, model, optimizer, scheduler=None, num_epoch=10, validation=False, use_gpu=False):
    '''
    Train the given model through the epochs.
    if validation is false, should be training mode
    '''
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    criterion=torch.nn.CrossEntropyLoss()
    model.train(not validation)
    five_crop = dataset.meta['five_crop']

    mode = 'Val' if validation else 'Train'

    for e in range(num_epoch):
        print('{} - Epoch {}..'.format(mode, e))
        epoch_start = time.clock()
        running_loss = 0.0
        running_corrects = 0 
        if scheduler is not None: scheduler.step()
        for data in loader:
            optimizer.zero_grad()
            inputs, labels = data['image'], data['label']
            if five_crop:
                # Handling 5 crop by unfolding
                _, _, channel, size, _ = inputs.size()
                inputs = inputs.reshape(-1, channel, size, size)

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:    
                inputs, labels = Variable(inputs), Variable(labels)
            outputs = model.forward(inputs)

            if five_crop:
                # handle averaging of the 5 crop prediction
                outputs = outputs.reshape(-1, 5, len(dataset.classes)).mean(dim=1)

            _, predictions = outputs.max(dim=1)
            loss = criterion(outputs, labels) / len(dataset)
            if not validation:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            running_corrects += (predictions.cpu() == labels.cpu()).sum().item()
        
        epoch_loss = running_loss 
        epoch_acc = running_corrects / float(len(dataset)) 

        epoch_time = time.clock() - epoch_start
        print("      >> Epoch loss {:.5f} accuracy {:.3f}        \
              in {:.4f}s".format(epoch_loss, epoch_acc, epoch_time))

    return model


# some test
def test_dataset(dataset, index=0):
    # testing dataset getitem
    ww = dataset[index]
    print('dataset length', len(dataset), ww['image'].size())

# setup & run training
def run_validation(five_crop=False, dataset_count=250, use_gpu=True):
    '''
    Run training with preset parameters.
    '''
    wk3dataset = Wk3Dataset('../datasets/imagenet_first2500/', data_limit=dataset_count, five_crop=five_crop)
    test_dataset(wk3dataset)
    # return

    model_ft = models.resnet18(pretrained=True)
    if use_gpu:
        model_ft = model_ft.cuda(0)
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

    model_ft = train_model(wk3dataset, model_ft, optimizer_ft, 
                           num_epoch=1, validation=True, use_gpu=use_gpu)
    return wk3dataset, model_ft

def main():
    run_validation(five_crop=False, dataset_count=160)
    run_validation(five_crop=True, dataset_count=160)

if __name__ == '__main__':
    main()
    
