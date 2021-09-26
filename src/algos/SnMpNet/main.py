import sys
import numpy as np
import torch

sys.path.append('/home/soumava/UCDR/src/')
# user defined
from trainer import Trainer
from options.options_snmpnet import Options


def main(args):

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    print('\nDevice:{}'.format(device))

    trainer = Trainer(args)
    trainer.do_training()


if __name__ == '__main__':

    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))
    main(args)