'''
Support code for Homework 6
LSTM - Star Trek Script
'''
import os.path, time
import unicodedata, string
import glob, random 

import torch
import pickle

'''
-------- Helper Function ------
'''
charspace = string.ascii_letters + " .,;'-\n" # use \n as EOL
def letter_index(letter): 
    return charspace.find(letter)

# find all files in path
def find(path): 
    return glob.glob(path)

# strip unacceptable symbols
def strip_symbols(s, to_space='=/+()[]'):
    for t in to_space:
        s = s.replace(t, ' ')
    return s

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

# split csv according to `,` but preserving any `, ` which is usually in a line
def split_csv(s, ignore=', '):
    # uses '$' to replace `, `, and ^ to replace','
    # and then split by '^' and convert '$' back to ', '
    s = s.replace(ignore, '$')
    s = s.replace(',', '^')
    s = s.replace('$', ignore)
    return s.split('^')
    
'''
-------- Dataset ---------
'''
from torch.utils.data import Dataset, DataLoader
# make it a dataset class
class MovieScriptDataset(Dataset):
    def __init__(self, root_path, ext='.csv', filterwords=[], shuffle=True):
        self.meta = {"root_path": root_path,
                     "ext": ext,
                     "filterwords": filterwords,
                     "shuffle": shuffle}
        self.line_list = []
        if root_path is not None:
            self.read_script()
        if shuffle: self.shuffle()

    def shuffle(self):
        random.shuffle(self.line_list)

    def read_script(self):
        '''
        Read the script given in root_path.
        '''
        filename = self.meta['root_path']
        with open(filename, 'r') as infile:
            filecontent = strip_symbols(infile.read())
            if self.meta['ext'] == '.csv':
                filecontent = split_csv(filecontent)
            else:
                filecontent = filecontent.split('\n')
            lines = [convert_to_ascii(content.strip()) for content in filecontent
                     if len(content.strip()) > 1 and content.strip() not in self.meta['filterwords']]
        self.line_list += lines
            
    def split_train_test(self, train_fraction=0.7):
        '''
        Creates 2 new MovieScriptDataset and fill them
        with 70/30 of own data.
        '''
        midline = int(len(self.line_list) * train_fraction)
        train = MovieScriptDataset(None)
        train.meta = self.meta
        train.line_list = self.line_list[:midline]

        test = MovieScriptDataset(None)
        test.meta = self.meta
        test.line_list = self.line_list[midline:]
        return train, test

    def __len__(self):
        return len(self.line_list)

    def __getitem__(self, index):
        '''
        Take the name-language pair and convert them to tensors
        '''
        line = self.line_list[index]
        line_tensor = string_to_tensor(line)
        # since the goal is to predict the next letter,
        # the label should be the line shifted, plus EOL (\n)
        label = line[1:] + '\n'
        label_tensor = torch.Tensor([charspace.index(l) for l in label])

        return {'input': line_tensor,
                'label': label_tensor,
                'seq_len': len(line)}

    # @staticmethod
    # def collate_function():
    #     '''
    #     Gives a specific collate_fn to be used with this dataset
    #     '''
    #     return collator

# must be executed at top-level, otherwise cant pickle
def MovieScriptCollator(tensorlist):
    # tensorlist: a list of tensors from __getitem__
    tensorlist = sorted(tensorlist, key=lambda x: x['seq_len'], reverse=True)
    data_tensor = rnn_utils.pack_sequence([val['input'] for val in tensorlist])
    label_tensor = rnn_utils.pack_sequence([val['label'] for val in tensorlist])
    return {'input': data_tensor,
            'label': label_tensor,
            'batch_size': len(tensorlist)}


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
        self.stack_fc = nn.Linear(hidden_size, hidden_size)
        self.fc_dropout = nn.Dropout(0.1)
        self.cover_fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, inputs, cache):
        '''
        cache = (hidden0, cell0) in a tuple
        '''
        output, (hn, cn) = super(CoveredLSTM, self).forward(inputs, cache)
        # print('lstm output', output.data.size())
        # print('lstm batchsize', output.batch_sizes, 'size', output.batch_sizes.size())
        # to_stack = [output]
        # for i in range(self.num_layers):
        #     to_stack.append(hn[i:i+1])
        # output_combined = torch.cat(to_stack, 2)
        # output_combined = output.view()
        if isinstance(output, rnn_utils.PackedSequence):
            stack_output = self.stack_fc(output.data)
            dropped_stack_output = self.fc_dropout(stack_output)
            covered_output = self.cover_fc(dropped_stack_output) 
            return covered_output, output.batch_sizes
        else:
            stack_output = self.stack_fc(output)
            dropped_stack_output = self.fc_dropout(stack_output)
            covered_output = self.cover_fc(dropped_stack_output) 
            return covered_output, None

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

    def sample(self, start_letter=None, max_length=70, use_gpu=True):
        '''
        With the current model, get a sample line.
        category should already be parsed (an integer/tensor)
        '''
        if start_letter == None:
            # start_letter = random.choice(string.ascii_letters)
            start_letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        self.train() # put in train mode to keep randomization

        with torch.no_grad():
            inputs = string_to_tensor(start_letter).view(1,1,-1)
            cache = self.init_cache()
            output_line = start_letter

            for i in range(max_length):
                if use_gpu: inputs = inputs.cuda()
                output, cache = self.forward(inputs, cache)
                _, gen = output.topk(1)

                # check EOL
                if int(gen.item()) >= len(charspace) - 1: # meaning \n
                    break # EOL reached - stop
                else:
                    new_letter = charspace[int(gen.item())]
                    output_line += new_letter
                    inputs = string_to_tensor(new_letter).view(1,1,-1)
            return output_line

    def save_model(self, filename):
        '''
        Save the model parameters as a pickled object.
        '''
        with open(filename, 'wb') as outfile:
            pickle.dump(self.state_dict(), outfile)

    def load_model(self, filename):
        '''
        Load model parameters from specified file.
        '''
        with open(filename, 'rb') as infile:
            loaded_dict = pickle.load(infile)
            self.load_state_dict(loaded_dict)


'''
Training 
'''
from torch.autograd import Variable
import torch.optim as optim

def train(dataset, model, batch_size=8, use_gpu=True, mode='train', lr=5e-2,
          epoch=1, print_every=1, sample_every=160):
    loader = DataLoader(dataset, batch_size=batch_size, 
                        collate_fn=MovieScriptCollator, num_workers=4)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss() 
    # criterion = nn.NLLLoss()
    model.train(mode == 'train')
    model.zero_grad()
    total_iter_count = (len(dataset) // batch_size) + 1

    for e in range(epoch):
        running_loss = 0.0
        running_corrects = 0
        total_letters = 0
        epoch_start = time.clock()

        iterr = 0
        for data in loader:
            iterr += 1
            optimizer.zero_grad()
            inputs, labels = data['input'], data['label']
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            cache = model.init_cache(batch=data['batch_size'], use_gpu=use_gpu)
            # from MovieScriptCollator, each batch has a 'batch size' to handle end cases 
            # e.g. if the last batch is less than batch_size

            model.zero_grad()
            output, _ = model(inputs, cache)
            _, pred = output.topk(1)

            loss = criterion(output, labels.data.long())
            running_loss += loss.item()
            if mode == 'train':
                loss.backward()
                optimizer.step()

            running_corrects += (pred.view(-1) == labels.data.long()).sum().item()
            total_letters += labels.data.size()[0]

            if (iterr % print_every) == 0:
                print('      ...iteration {}/{}'.format(iterr, total_iter_count), end='\r')
            if (iterr % sample_every) == 0:
                epoch_time = time.clock() - epoch_start
                print('      generated_sample:', model.sample(), 
                      '[loss:{:.5f} || acc:{:.3f} || in {:.4f}s]' \
                      .format(running_loss, running_corrects/total_letters, epoch_time))
            
        epoch_time = time.clock() - epoch_start
        print("      >> Epoch loss {:.5f} accuracy {:.3f}        \
              in {:.4f}s".format(running_loss, running_corrects/total_letters, epoch_time))
        model.save_model("trekmodel_e{}.clstm".format(e))
        print('save_model finished')
        sample_filename = "treksample_e{}.txt".format(e)
        with open(sample_filename, 'w') as samplefile:
            for i in range(20):
                samplefile.write(model.sample() + '\n')
        print('save_sample finished')
        
    return model, running_loss, running_corrects

def main():
    # split_csv test
    cssv = "I am groot.,Groot, I am."
    print(split_csv(cssv))

    star_filter = ['NEXTEPISODE']
    dataset = MovieScriptDataset('../dataset/startrek/star_trek_transcripts_all_episodes_f.csv',
                                 filterwords=star_filter)
    # dataset, _ = dataset.split_train_test(train_fraction=0.001) # getting smaller data
    train_data, test_data = dataset.split_train_test()

    lstm_mod = CoveredLSTM(len(charspace), 200, 3, len(charspace)).cuda()

    print("training")
    trained_model, loss, acc = train(train_data, lstm_mod, lr=5e-2, batch_size=4, mode='train', epoch=5)
    print("On testing data")
    final_model, loss, acc = train(test_data, trained_model, lr=5e-2, batch_size=4, mode='test')

if __name__ == '__main__':
    main()

