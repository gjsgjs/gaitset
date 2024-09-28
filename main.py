import argparse
from model.initialization import initialization
from config import conf

parser = argparse.ArgumentParser(description='Main program for opengait.')
parser.add_argument('--phase', default='train', choices=['train', 'test'], help="choose train or test phase")
parser.add_argument('--iter', default=0, help="iter to restore")
parser.add_argument('--batch_size', default='1', type=int,
                    help='batch_size: batch size for parallel test. Default: 1')
opt = parser.parse_args()

if __name__ == '__main__':
    training = (opt.phase == 'train')
    conf['model']['restore_iter'] = opt.iter

    m = initialization(conf, training)[0]
    if training:
        m.run_train()
    else:
        m.run_test(batch_size=opt.batch_size)