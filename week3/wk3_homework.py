
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import PIL.Image as Image

import torchvision
from torchvision import models, transforms, utils

import getimagenetclasses as ginc
import os
import math
import numpy as np


#####
#   Dataset subclass
#####

class Wk3Dataset(Dataset):
    def __init__(self, root_dir, file_prefix='ILSVRC2012_val_',
                 img_ext='.JPEG', val_ext='.xml', synset='synset_words.txt',
                 five_crop=False, data_limit=0):
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
        i2s, s2i, s2d = ginc.parsesynsetwords(self.meta['synset'])
        self.dataset = [filename[len(file_prefix):-len(img_ext)]
                        for filename in os.listdir(os.path.join(root_dir, 'imagespart'))]
        if data_limit > 0:
            self.dataset = self.dataset[:data_limit]
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
        label, firstname = ginc.parseclasslabel(self.get_val_path(index), self._rev_dataset)
        # convert label to a vector of size len_class
        label_vector = np.zeros(len(self.classes))
        label_vector[int(label)] = 1
        
        # 2. load the image file
        image = Image.open(self.get_img_path(index))
        image = self.transform_short(image)

        five_crop = self.meta['five_crop']

        if five_crop:
            image = self.transform_fivecrop(image)
        else:
            image = self.transform_centercrop(image)

        image = transforms.ToTensor()(image)

        # dealing with grayscale - centercrop
        if not five_crop and image.size()[0] == 1:
            image = image.repeat([3,1,1])

        # dealing with grayscale - fivecrop
        if five_crop and image.size()[0] == 1:
            image = image.repeat([3,1,1,1])

        itm = {'label':label_vector.astype(np.float32),
               #'firstname': firstname,
               'image':image,
               #'filename':self.get_img_path(index)
               }
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
        assert NotImplementedError


######
#   Training codes
######

def train_model(dataset, model, optimizer, num_epoch=10, validation=False, use_gpu=False):
    '''
    Train the given model through the epochs.
    if validation is false, should be training mode
    '''
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)
    criterion=torch.nn.BCEWithLogitsLoss(weight=None)
    model.train(not validation)
    for e in range(num_epoch):
        print('Epoch {}..'.format(e))
        # TODO: implement AlexNet (should be simple ones)
        running_loss = 0
        for data in loader:
            inputs, labels = data['image'], data['label']
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            outputs = model(inputs)
            #predictions = outputs.data >= 0

            loss=0
            for c in range(len(dataset.classes)):
                loss += criterion(outputs[:,c], labels[:,c])

            if not validation:
                loss.backward()
                optimizer.step()

            running_loss += loss.cpu().data[0]
        
        epoch_loss = running_loss / len(dataset)
        print("Epoch loss: {}".format(epoch_loss))

    return model


# some test

def test_dataset(dataset, index=10):
    # testing dataset getitem
    ww = dataset[index]
    print('dataset length', len(dataset))
    # print('loaded label', dataset.dataset[index], ww['label'], ww['image'].size())

def run_training():
    '''
    Run training with preset parameters.
    '''
    wk3dataset = Wk3Dataset('../datasets/imagenet_first2500/', 
                            data_limit=160) # put data_limit=0 for loading everything
    test_dataset(wk3dataset)
    # OK up to this point

    #model - copied straight from pascalvoc
    # TODO: revise / modify
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(wk3dataset.classes))
    
    # optimizer - straight from pascalvoc
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
    model_ft = train_model(wk3dataset, model_ft, optimizer_ft, num_epoch=1)


def main():
    run_training()

if __name__ == '__main__':
    main()
    
