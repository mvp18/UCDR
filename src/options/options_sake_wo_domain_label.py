"""
    Parse input arguments
"""
import argparse


class Options:

    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='SAKE w/o Domain Label for ZS-SBIR.')
        
        parser.add_argument('-root', '--root_path', default='/data/soumava/datasets/', type=str)
        parser.add_argument('-path_cp', '--checkpoint_path', default='/data/soumava/saved_models/SAKE_wo_domain_label/', type=str)
        parser.add_argument('-resume', '--resume_dict', default=None, type=str, help='checkpoint file to resume training from')

        parser.add_argument('-data', '--dataset', default='Sketchy', choices=['Sketchy', 'TUBerlin'])
        parser.add_argument('-eccv', '--is_eccv_split', choices=[0, 1], default=1, type=int, help='whether or not to use eccv18 split\
                            if dataset="Sketchy"')

        # Loss weight & reg. parameters
        parser.add_argument('-kdl', '--kd_lambda', metavar='LAMBDA', default=1.0, type=float, help='lambda for kd loss (default: 1)')
        parser.add_argument('-kdnl', '--kdneg_lambda', metavar='LAMBDA', default=0.3, type=float, help='lambda for semantic adjustment (default: 0.3)')
        parser.add_argument('-sl', '--sake_lambda', metavar='LAMBDA', default=1.0, type=float, help='lambda for total SAKE loss (default: 1)')
        parser.add_argument('-l2', '--l2_reg', default=5e-4, type=float, help='L2 Weight Decay for optimizer')
        parser.add_argument('-ems', '--ems', choices=[0, 1], default=1, type=int, help='to use simple cce/Euclidean Margin Softmax loss for training.\
                            In the latter case, model has an EMS layer instead of simple linear.')

        # Size parameters
        parser.add_argument('-imsz', '--image_size', default=224, type=int, help='Input size for query/gallery domain sample')
        parser.add_argument('-hd', '--hashing_dim', default=512, type=int, help='Feature Dimension for retrieval')
        
        # Model parameters
        parser.add_argument('-seed', '--seed', type=int, default=0)
        parser.add_argument('-bs', '--batch_size', default=40, type=int)
        parser.add_argument('-nw', '--num_workers', type=int, default=4, help='Number of workers in data loader')
        
        # Optimization parameters
        parser.add_argument('-e', '--epochs', type=int, default=20, metavar='N', help='Number of epochs to train (default: 100)')
        parser.add_argument('-lr', '--lr', type=float, default=0.0001, metavar='LR', help='Initial learning rate for optimizer & scheduler')

        # Checkpoint parameters
        parser.add_argument('-es', '--early_stop', type=int, default=30, help='Early stopping epochs.')

        # I/O parameters
        parser.add_argument('-log', '--log_interval', type=int, default=100, metavar='N', help='How many batches to wait before logging training status')

        self.parser = parser

    
    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()