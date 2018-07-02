'''
Plotter only
'''

from hw6_code import *

def main():
    with open('trekstats.stt', 'rb') as statfile:
        data = pickle.load(statfile)
        train_loss_acc = data['train']
        print(train_loss_acc)
        test_loss_acc  = data['test']
        print(test_loss_acc)
        plot_over_epoch(train_loss_acc, test_loss_acc)

if __name__ == '__main__':
    main()

    