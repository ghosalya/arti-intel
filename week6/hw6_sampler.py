'''
Sampler only
- loads from latest model
'''
from hw6_code import *

def main():
    lstm_mod = CoveredLSTM(len(charspace), 200, 2, len(charspace)).cuda()
    lstm_mod.load_model('models/trekmodel_latest.clstm')

    for t in [2, 1, 0.75, 0.5]:
        print('for temp {}'.format(t))
        for i in range(3):
            print('     ', lstm_mod.sample(temperature=t))

if __name__ == '__main__':
    main()