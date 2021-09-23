"""
    Parse input arguments
"""
import argparse


class Options:

    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='Doodle2Search for ZS-SBIR')
        
        parser.add_argument('-root', '--root_path', default='/data/soumava/datasets/', type=str)
        parser.add_argument('-path_cp', '--checkpoint_path', default='/data/soumava/saved_models/Base_NW/', type=str)
        parser.add_argument('-resume', '--resume_dict', default=None, type=str, help='checkpoint file to resume training from')

        parser.add_argument('-data', '--dataset', default='DomainNet', choices=['Sketchy', 'DomainNet'])

        # Loss weight & reg. parameters
        parser.add_argument('-grl', '--grl_lambda', type=float, default=0.5, help='Lambda used to normalize the GRL layer.')
        parser.add_argument('-wsem', '--w_semantic', default=1.0, type=float, help='Semantic loss Weight.')
        parser.add_argument('-wdom', '--w_domain', default=1.0, type=float, help='Domain loss Weight.')
        parser.add_argument('-wtrip', '--w_triplet', type=float, default=1.0, help='Triplet loss Weight.')
        parser.add_argument('-l2', '--l2_reg', default=5e-4, type=float, help='L2 Weight Decay for optimizer')

        # Size parameters
        parser.add_argument('-semsz', '--semantic_emb_size', choices=[200, 300], default=300, type=int, help='Glove vector dimension')
        parser.add_argument('-imsz', '--image_size', default=224, type=int, help='Input size for query/gallery domain sample')
        
        # Model parameters
        parser.add_argument('-arch', '--architecture', choices=['vgg16', 'seresnet50'], default='vgg16')
        parser.add_argument('-seed', '--seed', type=int, default=0)
        parser.add_argument('-bs', '--batch_size', default=64, type=int)
        parser.add_argument('-nw', '--num_workers', type=int, default=4, help='Number of workers in data loader')
        
        # Optimization parameters
        parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N', help='Number of epochs to train')
        parser.add_argument('-lr', '--lr', type=float, default=1e-4, metavar='LR', help='Initial learning rate for optimizer & scheduler')
        parser.add_argument('-mom', '--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')

        # Checkpoint parameters
        parser.add_argument('-es', '--early_stop', type=int, default=30, help='Early stopping epochs.')

        # I/O parameters
        parser.add_argument('-log', '--log_interval', type=int, default=400, metavar='N', help='How many batches to wait before logging training status')

        self.parser = parser

    
    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()