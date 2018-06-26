'''
Support code for HW5 RNN part
'''
import os.path, time
import unicodedata, string
import glob, random 

import torch

'''
-------- Helper Function ------
'''
charspace = string.ascii_letters + ".,;'"
def letter_index(letter): 
    return charspace.find(letter)

# find all files in path
def find(path): 
    return glob.glob(path)

# turn unicode string to ascii
def convert_to_ascii(unicode_strings):
    return ''.join(c for c in unicodedata.normalize('NFD', unicode_strings)
                    if unicodedata.category(c) != 'Mn' and c in charspace)

# convert string to a one-hot tensor
def string_to_tensor(inputstring, pad_to=0):
    # pad_to: pad result to a certain length
    tensor = torch.zeros(max(len(inputstring), pad_to), 
                         len(charspace))
    for i, letter in enumerate(inputstring):
        tensor[i][letter_index(letter)] = 1
    return tensor

'''
-------- Dataset ---------
'''
from torch.utils.data import Dataset, DataLoader
# make it a dataset class
class InternationalNameDataset(Dataset):
    def __init__(self, root_path, ext='.txt', shuffle=True, padded=True):
        self.meta = {"root_path": root_path,
                     "ext": ext,
                     "shuffle": shuffle,
                     "padded": padded}
        if root_path is None:
            self.name_list = None
            self.categories = None
            if padded: self.meta["pad_maxsize"] = 0
        else:
            categories, name_list = self.get_namelist()
            self.name_list = name_list
            self.categories = categories
            if shuffle: self.shuffle()
            if padded: self.meta["pad_maxsize"] = self.get_pad_maxsize()

    def get_namelist(self):
        path = os.path.join(self.meta['root_path'], "*" + self.meta['ext'])
        categories = []
        name_list = [] # list of names in a language i.e. {'lang':'name'}
        for filename in find(path): # iterate through file names
            category = os.path.split(filename)[-1].split('.')[0]
            categories.append(category)
            with open(filename, 'r', encoding='utf-8') as infile:
                # read the file content to a list
                lines = infile.read().strip().split('\n')
                lines = [convert_to_ascii(line) for line in lines]
            # add the list of line-language pair to name_list
            name_list += [(line, category) for line in lines]
        return categories, name_list

    def shuffle(self):
        random.shuffle(self.name_list)

    def get_pad_maxsize(self):
        maxx = 0
        for line, lang in self.name_list:
            if maxx < len(line): maxx = len(line)
        return maxx

    def split_train_test(self, train_fraction=0.7):
        '''
        Creates 2 new InternationalNameDataset and fill them
        with 70/30 of own data.
        '''
        midline = int(len(self.name_list) * train_fraction)
        train = InternationalNameDataset(None)
        train.meta = self.meta
        train.categories = self.categories
        train.name_list = self.name_list[:midline]

        test = InternationalNameDataset(None)
        test.meta = self.meta
        test.categories = self.categories
        test.name_list = self.name_list[midline:]
        return train, test

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        '''
        Take the name-language pair and convert them to tensors
        '''
        line, category = self.name_list[index]
        category_index = self.categories.index(category)
        category_onehot = torch.zeros(len(self.categories))
        category_onehot[category_index] = 1
        line_tensor = string_to_tensor(line, pad_to=self.meta["pad_maxsize"])
        return {'label': category_index,
                'input': line_tensor,
                'seq_len': len(line)}

'''
--------   RNN/LSTM   ---------
'''
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
class CoveredLSTM(nn.LSTM):
    '''
    A stack of LSTM that is covered by a fully-connected layer
    as the last layer. The fully-connected layer is fed the hidden
    state of the last LSTM stack (if any). CrossEntropyLoss should be 
    used as the output is a tensor of length num_class.
    '''
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CoveredLSTM, self).__init__(input_size, hidden_size, num_layers)
        self.num_layers = num_layers
        self.cover_fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, inputs, cache):
        '''
        cache = (hidden0, cell0) in a tuple
        '''
        output, (hn, cn) = super(CoveredLSTM, self).forward(inputs, cache)
        covered_output = self.cover_fc.forward(hn[-1]) # taking the last lstm's hn
        return covered_output, (output, hn, cn)    

    def init_cache(self, batch=1, use_gpu = True):
        '''
        "batch" parameter is added in case a
        stacked multiple hidden matrix is needed
        (e.g. for multibatch forward pass)
        '''
        h0 = torch.zeros(self.num_layers, batch, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch, self.hidden_size)
        if use_gpu:
            h0, c0 = h0.cuda(), c0.cuda()
        return (h0, c0)

class RNNSampler(object):
    '''
    Sampler for RNN/LSTM training
    It will generate a batch list (of index)
    and sort them by the sequence length
    '''
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        batch = []
        for d in range(len(self.dataset)):
            batch.append(d)
            if len(batch) >= self.batch_size or d+1 == len(self.dataset):
                # sort to decreasing length
                sorted_batch = sorted(batch, reverse=True,
                                      key=lambda x: self.dataset[x]['seq_len'])
                yield sorted_batch
                batch = []

'''
Training 
'''
from torch.autograd import Variable
import torch.optim as optim

def train(dataset, model, batch_size=8, use_gpu=True, mode='train', lr=5e-2,
          epoch=1, print_every=100):
    loader = DataLoader(dataset, batch_sampler=RNNSampler(dataset, batch_size), num_workers=4)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss() 
    model.train(mode == 'train')
    model.zero_grad()
    total_iter_count = (len(dataset) // batch_size) + 1

    for e in range(epoch):
        running_loss = 0.0
        running_corrects = 0
        epoch_start = time.clock()

        iterr = 0
        for data in loader:
            iterr += 1
            optimizer.zero_grad()
            inputs = data['input'].transpose(0,1) 
            # transposing to match LSTM formatting
            labels = data['label']
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # packing padded sequence
            padded_inputs = rnn_utils.pack_padded_sequence(inputs, data['seq_len'])
            cache = model.init_cache(batch=inputs.size()[1], use_gpu=use_gpu)
            model.zero_grad()

            output, cache = model(padded_inputs, cache)
            _, pred = output.max(dim=1)
            
            loss = criterion(output, labels.long())
            if mode == 'train':
                loss.backward()
                optimizer.step()

            running_loss += loss.item() / float(len(dataset))
            running_corrects += (pred == labels.long()).sum().item()/float(len(dataset))

            if (iterr % print_every) == 0:
                print('      ...iteration {}/{}'.format(iterr, total_iter_count), end='\r')
            
        epoch_time = time.clock() - epoch_start
        print("      >> Epoch loss {:.5f} accuracy {:.3f}        \
              in {:.4f}s".format(running_loss, running_corrects, epoch_time))
    return model, running_loss, running_corrects

'''
----- Plotting
'''
import matplotlib.pyplot as plt 

def plot_varying_epochs(loss_acc_dict):
    plt.figure()
    train_losses = [loss_acc_dict[i][0] for i in sorted(loss_acc_dict.keys())]
    test_losses = [loss_acc_dict[i][1] for i in sorted(loss_acc_dict.keys())]
    accuracy = [loss_acc_dict[i][2] for i in sorted(loss_acc_dict.keys())]
    
    plt.subplot(131)
    plt.plot(train_losses, color='purple')
    plt.xticks(range(4), [1,2,5,10])
    plt.xlabel('Epoch')
    plt.title("train_loss")

    plt.subplot(132)
    plt.plot(test_losses, color='gray')
    plt.xticks(range(4), [1,2,5,10])
    plt.xlabel('Epoch')
    plt.title("test_loss")

    plt.subplot(133)
    plt.plot(accuracy, color='blue')
    plt.xticks(range(4), [1,2,5,10])
    plt.xlabel('Epoch')
    plt.title("accuracy")

def main():
    dataset = InternationalNameDataset('../datasets/simplelanguage/data/names/')
    train_data, test_data = dataset.split_train_test()

    train_sampler = RNNSampler(train_data, 8)
    test_sampler = RNNSampler(test_data, 8)
    lstm_mod = CoveredLSTM(len(charspace), 128, 2, len(dataset.categories)).cuda()
    print("training")
    trained_model = train(train_data, lstm_mod, lr=5e-2, batch_size=8, mode='train')
    print("On testing data")
    final_model = train(test_data, lstm_mod, lr=5e-2, batch_size=8, mode='test')


if __name__ == '__main__':
    main()
