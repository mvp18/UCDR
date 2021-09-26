import sys
import os
import time
import numpy as np
import pickle
import glob
from datetime import datetime

# pytorch, torch vision
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.append('/home/soumava/UCDR/src/')
from options.options_snmpnet import Options
from data.Sketchy import sketchy_extended
from data.TUBerlin import tuberlin_extended
from data.DomainNet import domainnet
from data.dataloaders import BaselineDataset
from models.snmpnet.snmpnet import SnMpNet
from trainer import evaluate


def main(args):

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	use_gpu = torch.cuda.is_available()

	if use_gpu:
		cudnn.benchmark = True
		torch.cuda.manual_seed_all(args.seed)

	device = torch.device("cuda:0" if use_gpu else "cpu")
	print('\nDevice:{}'.format(device))

	if args.dataset=='Sketchy':
		data_input = sketchy_extended.create_trvalte_splits(args)

	if args.dataset=='DomainNet':
		data_input = domainnet.create_trvalte_splits(args)

	if args.dataset=='TUBerlin':
		data_input = tuberlin_extended.create_trvalte_splits(args)

	tr_classes = data_input['tr_classes']
	va_classes = data_input['va_classes']
	te_classes = data_input['te_classes']

	# Imagenet standards
	im_mean = [0.485, 0.456, 0.406]
	im_std = [0.229, 0.224, 0.225]

	# Image transformations
	image_transforms = {
		'train':
		transforms.Compose([
			transforms.RandomResizedCrop((args.image_size, args.image_size), (0.8, 1.0)),
			transforms.RandomHorizontalFlip(0.5),
			transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
			transforms.ToTensor(),
			transforms.Normalize(im_mean, im_std)
		]),

		'eval':
		transforms.Compose([
			transforms.Resize((args.image_size, args.image_size)),
			transforms.ToTensor(),
			transforms.Normalize(im_mean, im_std)
		]),
	}

	# Model
	model = SnMpNet(semantic_dim=args.semantic_emb_size, pretrained=None, num_tr_classes=len(tr_classes)).cuda()

	if args.dataset=='DomainNet':
		save_folder_name = 'seen-'+args.seen_domain+'_unseen-'+args.holdout_domain+'_x_'+args.gallery_domain
		if not args.include_auxillary_domains:
			save_folder_name += '_noaux'
	if args.dataset=='Sketchy':
		if args.is_eccv_split:
			save_folder_name = 'eccv_split'
		else:
			save_folder_name = 'random_split'
	else:
		save_folder_name = ''

	path_cp = os.path.join(args.checkpoint_path, args.dataset, save_folder_name)

	best_model_name = 'val_map200-0.7603_prec200-0.7333_ep-1_mixlevel-img_wcce-1.0_wratio-0.5_wmse-1.0_clswts-2.0_e-100_es-15_opt-sgd_bs-64_lr-0.001_l2-0.0_beta-1_seed-0_tv-0.pth'
	best_model_file = os.path.join(path_cp, best_model_name)

	if os.path.isfile(best_model_file):
				
		print("\nLoading best model from '{}'".format(best_model_file))
		# load the best model yet
		checkpoint = torch.load(best_model_file)
		epoch = checkpoint['epoch']
		best_map = checkpoint['best_map']
		model.load_state_dict(checkpoint['model_state_dict'])
		print("Loaded best model '{0}' (epoch {1}; mAP {2:.4f})\n".format(best_model_file, epoch, best_map))

		outstr = ''

		if args.dataset=='DomainNet':
			
			for domain in [args.seen_domain, args.holdout_domain]:
				for gzs in [0, 1]:

					test_head_str = 'Query:' + domain + '; Gallery:' + args.gallery_domain + '; Generalized:' + str(gzs)
					print(test_head_str)
					outstr += test_head_str

					splits_query = domainnet.trvalte_per_domain(args, domain, 0, tr_classes, va_classes, te_classes)
					splits_gallery = domainnet.trvalte_per_domain(args, args.gallery_domain, gzs, tr_classes, va_classes, te_classes)
					
					data_te_query = BaselineDataset(np.array(splits_query['te']), transforms=image_transforms['eval'])
					data_te_gallery = BaselineDataset(np.array(splits_gallery['te']), transforms=image_transforms['eval'])

					# PyTorch test loader for query
					te_loader_query = DataLoader(dataset=data_te_query, batch_size=64*5, shuffle=False, 
												 num_workers=args.num_workers, pin_memory=True)
					# PyTorch test loader for gallery
					te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=64*5, shuffle=False, 
												   num_workers=args.num_workers, pin_memory=True)

					print(f'#Test queries:{len(te_loader_query.dataset)}; #Test gallery samples:{len(te_loader_gallery.dataset)}.\n')
					te_data = evaluate(te_loader_query, te_loader_gallery, model, None, epoch, args)
				
					outstr+="\n\nmAP@200 = %.4f, Prec@200 = %.4f, mAP@all = %.4f, Prec@100 = %.4f"%(np.mean(te_data['aps@200']), te_data['prec@200'], 
							np.mean(te_data['aps@all']), te_data['prec@100'])

					outstr += '\n\n'
		else:
			data_splits = data_input['splits']
			data_te_query = BaselineDataset(data_splits['query_te'], transforms=image_transforms['eval'])
			data_te_gallery = BaselineDataset(data_splits['gallery_te'], transforms=image_transforms['eval'])

			te_loader_query = DataLoader(dataset=data_te_query, batch_size=args.batch_size*5, shuffle=False, num_workers=args.num_workers, pin_memory=True)
			te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=args.batch_size*5, shuffle=False, num_workers=args.num_workers, pin_memory=True)

			print(f'#Test queries:{len(te_loader_query.dataset)}; #Test gallery samples:{len(te_loader_gallery.dataset)}.\n')

			te_data = evaluate(te_loader_query, te_loader_gallery, model, None, epoch, args)
				
			outstr+="mAP@200 = %.4f, Prec@200 = %.4f, mAP@all = %.4f, Prec@100 = %.4f"%(np.mean(te_data['aps@200']), te_data['prec@200'], 
					np.mean(te_data['aps@all']), te_data['prec@100'])
		
		print(outstr)
		
	else:
		print(f'{best_model_file} not found!')


if __name__ == '__main__':
	# Parse options
	args = Options().parse()
	print('Parameters:\t' + str(args))
	main(args)