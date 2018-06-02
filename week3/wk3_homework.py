
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import PIL.Image as Image

import torchvision
from torchvision import models, transforms, utils

import getimagenetclasses as ginc
import os.path
import math

class Wk3Dataset(Dataset):
    def __init__(self, root_dir, file_prefix='ILSVRC2012_val_',
                 img_ext='.jpg', val_ext='xml', synset='synset_words.txt',
                 five_crop=False):
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

        if not five_crop:
            self.transform_crop = self.transform_centercrop
        else:
            self.transform_crop = self.transform_fivecrop

        # assertion for the mentioned assumption
        assert os.path.exists(os.path.join(root_dir, 'imagespart'))
        assert os.path.exists(os.path.join(root_dir, 'val'))

        # create list of synset_words if not found in directory
        if not os.path.exists(os.path.join(root_dir, synset)):
            self.generate_synset()

        # metadata
        self.classes = ginc.get_classes()
        i2s, s2i, s2d = ginc.parsesynsetwords(self.meta['synset'])
        self.dataset = [i2s[i] for i in range(len(i2s))]
        self._rev_dataset = s2i 
        self.data_description = s2d

    def get_val_path(self, index):
        zero_fill = math.floor(math.log10(len(self.dataset)))
        return os.path.join(self.meta['root_dir'], 'val',
                            self.meta['file_prefix'] + str(index).zfill(zero_fill) + self.meta['val_ext'])

    def get_img_path(self, index):
        zero_fill = math.floor(math.log10(len(self.dataset)))
        return os.path.join(self.meta['root_dir'], 'imagespart',
                            self.meta['file_prefix'] + str(index).zfill(zero_fill) + self.meta['img_ext'])
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        '''
        Only when __getitem__ is called should the
        code load the actual image
        '''
        # 1. get corresponding dataset metadata
        label, firstname = ginc.parseclasslabel(self.get_val_path(index), self.data_description)
        # 2. load the image file
        image = Image.open(self.get_img_path(index))
        image = self.transform_short(image)
        image = self.transform_crop(image)
        itm = {'label':label,
               'firstname': firstname,
               'image':image,
               'filename':self.get_img_path(index)}

    def generate_synset(self):
        pass

    def transform_short(self, image, short_size=280):
        '''
        Do the transformation:
        - resize till the shorter side is 280
        '''
        width, height = image.size
        ratio = short_size / min(width, height)
        new_size = (width, height) * ratio
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
        # TODO: 5 crop transformation
        pass


def train_model(dataloaders, model, optimizer, num_epoch=10, validation=False):
    '''
    Train the given model through the epochs.
    if validation is false, should be training mode
    '''
    criterion=torch.nn.BCEWithLogitsLoss(weight=None)
    model.train(not validation)
    for e in range(num_epoch):
        print('Epoch {}..'.format(e))
        # TODO: implement AlexNet (should be simple ones)

    return None


def run_training():
    '''
    Run training with preset parameters.
    '''
    wk3dataset = Wk3Dataset('../datasets/imagenet_first2500/')
    loader = DataLoader(wk3dataset, batch_size=32, shuffle=True, num_workers=4)

    #model - copied straight from pascalvoc
    # TODO: revise / modify
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(wk3dataset.classes))
    
    # optimizer - straight from pascalvoc
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

    model_ft = train_model(loader, model_ft, optimizer_ft)


def main():
    run_training()

if __name__ == '__main__':
    main()
    
