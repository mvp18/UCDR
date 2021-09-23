"""
    Parse input arguments
"""
import argparse


class Options:

    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='CuMix for UCDR')
        
        parser.add_argument('-root', '--root_path', default='/data/soumava/datasets/', type=str)
        parser.add_argument('-path_cp', '--checkpoint_path', default='/data/soumava/saved_models/CuMix/', type=str)
        parser.add_argument('-resume', '--resume_dict', default=None, type=str, help='checkpoint file to resume training from')

        parser.add_argument('-data', '--dataset', default='DomainNet', choices=['Sketchy', 'DomainNet'])
        
        # DomainNet specific arguments
        parser.add_argument('-sd', '--seen_domain', default='sketch', choices=['quickdraw', 'clipart', 'infograph', 'sketch', 'painting'])
        parser.add_argument('-hd', '--holdout_domain', default='quickdraw', choices=['quickdraw', 'clipart', 'infograph', 'sketch', 'painting'])
        parser.add_argument('-gd', '--gallery_domain', default='real', choices=['clipart', 'infograph', 'photo', 'painting', 'real'])
        parser.add_argument('-aux', '--include_auxillary_domains', choices=[0, 1], default=1, type=int, help='whether(1) or not(0) to include\
                            domains other than seen domain and gallery')

        parser.add_argument('-opt', '--optimizer', type=str, choices=['sgd', 'adam'], default='sgd')

        # Loss weight & reg. parameters
        parser.add_argument('-wcce', '--wcce', default=1.0, type=float, help='Weight on cosine-based cce loss')
        parser.add_argument('-wimg', '--mixup_w_img', type=float, default=0.001, help='Weight to image mixup CE loss')
        parser.add_argument('-wfeat', '--mixup_w_feat', type=float, default=2, help='Weight to feature mixup CE loss')
        parser.add_argument('-l2', '--l2_reg', default=5e-5, type=float, help='L2 Weight Decay for optimizer')

        # Size parameters
        parser.add_argument('-semsz', '--semantic_emb_size', choices=[200, 300], default=300, type=int, help='Glove vector dimension')
        parser.add_argument('-imsz', '--image_size', default=224, type=int, help='Input size for query/gallery domain sample')
        
        # Model parameters
        parser.add_argument('-seed', '--seed', type=int, default=0)
        parser.add_argument('-beta', '--mixup_beta', type=float, default=1, help='mixup interpolation coefficient')
        parser.add_argument('-step', '--mixup_step', type=int, default=2, help='Initial warmup steps for domain and class mixing ratios.')
        parser.add_argument('-bs', '--batch_size', default=40, type=int)
        parser.add_argument('-nw', '--num_workers', type=int, default=6, help='Number of workers in data loader')
        
        # Optimization parameters
        parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N', help='Number of epochs to train')
        parser.add_argument('-lr', '--lr', type=float, default=0.001, metavar='LR', help='Initial learning rate for optimizer & scheduler')
        parser.add_argument('-mom', '--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')

        # Checkpoint parameters
        parser.add_argument('-es', '--early_stop', type=int, default=30, help='Early stopping epochs.')

        # I/O parameters
        parser.add_argument('-log', '--log_interval', type=int, default=400, metavar='N', help='How many batches to wait before logging training status')

        self.parser = parser

    
    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()