import os
import time
import numpy as np
import math
from scipy.spatial.distance import cdist
import pickle
import paramiko

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from data.Sketchy import sketchy_extended
from data.TUBerlin import tuberlin_extended
from data.DomainNet import domainnet
from data.dataloaders import BaselineDataset, CuMixloader
from data.sampler import BalancedSampler
from models.snmpnet.snmpnet import SnMpNet
from losses.embedding_losses import Mixup_Cosine_CCE, Mixup_Euclidean_MSE
from utils import utils
from utils.metrics import compute_retrieval_metrics
from utils.logger import AverageMeter


class Trainer:
	
	def __init__(self, args):

		self.args = args

		print('\nLoading data...')

		if args.dataset=='Sketchy':
			data_input = sketchy_extended.create_trvalte_splits(args)

		if args.dataset=='DomainNet':
			data_input = domainnet.create_trvalte_splits(args)

		if args.dataset=='TUBerlin':
			data_input = tuberlin_extended.create_trvalte_splits(args)

		self.tr_classes = data_input['tr_classes']
		self.va_classes = data_input['va_classes']
		semantic_vec = data_input['semantic_vec']
		data_splits = data_input['splits']

		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		use_gpu = torch.cuda.is_available()

		if use_gpu:
			cudnn.benchmark = True
			torch.cuda.manual_seed_all(args.seed)
		
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

		# class dictionary
		self.dict_clss = utils.create_dict_texts(self.tr_classes)

		fls_tr = data_splits['tr']
		cls_tr = np.array([f.split('/')[-2] for f in fls_tr])
		dom_tr = np.array([f.split('/')[-3] for f in fls_tr])
		tr_domains_unique = np.unique(dom_tr)

		# doamin dictionary
		self.dict_doms = utils.create_dict_texts(tr_domains_unique)
		print(self.dict_doms)
		domain_ids = utils.numeric_classes(dom_tr, self.dict_doms)
		
		data_train = CuMixloader(fls_tr, cls_tr, dom_tr, self.dict_doms, transforms=image_transforms['train'])
		train_sampler = BalancedSampler(domain_ids, args.batch_size//len(tr_domains_unique), domains_per_batch=len(tr_domains_unique))
		self.train_loader = DataLoader(dataset=data_train, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, 
									   pin_memory=True)
		
		data_va_query = BaselineDataset(data_splits['query_va'], transforms=image_transforms['eval'])
		data_va_gallery = BaselineDataset(data_splits['gallery_va'], transforms=image_transforms['eval'])
		
		# PyTorch valid loader for query
		self.va_loader_query = DataLoader(dataset=data_va_query, batch_size=args.batch_size*5, shuffle=False, num_workers=args.num_workers, 
										  pin_memory=True)
		# PyTorch valid loader for gallery
		self.va_loader_gallery = DataLoader(dataset=data_va_gallery, batch_size=args.batch_size*5, shuffle=False, num_workers=args.num_workers, 
											pin_memory=True)
		
		print(f'#Tr samples:{len(data_train)}; #Val queries:{len(data_va_query)}; #Val gallery samples:{len(data_va_gallery)}.\n')
		print('Loading Done\n')

		# Model
		self.model = SnMpNet(semantic_dim=args.semantic_emb_size, pretrained='imagenet', num_tr_classes=len(self.tr_classes)).cuda()

		self.glove_embed_seen = np.array([semantic_vec.get(cl) for cl in self.tr_classes])
		self.retrieval_loss = Mixup_Cosine_CCE(torch.from_numpy(self.glove_embed_seen).float().cuda())
		self.embedding_loss = Mixup_Euclidean_MSE(torch.from_numpy(self.glove_embed_seen).float().cuda(), args.alpha)
		
		self.RG = np.random.default_rng()

		if args.optimizer=='sgd':
			self.optimizer = optim.SGD(self.model.parameters(), weight_decay=args.l2_reg, momentum=args.momentum, nesterov=False, lr=args.lr)
		elif args.optimizer=='adam':
			self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.l2_reg)
		
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

		if args.dataset=='DomainNet' or (args.dataset=='Sketchy' and args.is_eccv_split):
			self.map_metric = 'aps@200'
			self.prec_metric = 'prec@200'
		else:
			self.map_metric = 'aps@all'
			self.prec_metric = 'prec@100'
		
		self.path_cp = os.path.join(args.checkpoint_path, args.dataset, save_folder_name)
		
		self.suffix = '_mixlevel-'+args.mixup_level+'_wcce-'+str(args.wcce)+'_wratio-'+str(args.wratio)+'_wmse-'+str(args.wmse)+\
					  '_clswts-'+str(args.alpha)+'_e-'+str(args.epochs)+'_es-'+str(args.early_stop)+'_opt-'+args.optimizer+\
					  '_bs-'+str(args.batch_size)+'_lr-'+str(args.lr)+'_l2-'+str(args.l2_reg)+'_beta-'+str(args.mixup_beta)+\
					  '_warmup-'+str(args.mixup_step)+'_seed-'+str(args.seed)+'_tv-'+str(args.trainvalid)
		
		# exit(0)
		path_log = os.path.join('./logs', args.dataset, save_folder_name, self.suffix)
		# Logger
		print('Setting logger...', end='')
		self.logger = SummaryWriter(path_log)
		print('Done\n')

		self.start_epoch = 0
		self.best_map = 0
		self.early_stop_counter = 0
		self.last_chkpt_name='init'

		self.resume_from_checkpoint(args.resume_dict)
	
	
	def adjust_learning_rate(self, min_lr=1e-6):
		# lr = args.lr * 0.5 * (1.0 + math.cos(float(epoch) / args.epochs * math.pi))
		# epoch_curr = min(epoch, 20)
		# lr = args.lr * math.pow(0.001, float(epoch_curr)/ 20 )
		lr = self.args.lr * math.pow(1e-3, float(self.current_epoch)/20)
		lr = max(lr, min_lr)
		# print('epoch: {}, lr: {}'.format(epoch, lr))
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr


	def resume_from_checkpoint(self, resume_dict):

		if resume_dict is not None:
			print('==> Resuming from checkpoint: ',resume_dict)
			checkpoint = torch.load(os.path.join(self.path_cp, resume_dict+'.pth'))
			self.start_epoch = checkpoint['epoch']+1
			self.model.load_state_dict(checkpoint['model_state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			self.best_map = checkpoint['best_map']
			self.last_chkpt_name = resume_dict


	def swap(self, xs, a, b):
		xs[a], xs[b] = xs[b], xs[a]

	
	def derange(self, xs):
		x_new = [] + xs
		for a in range(1, len(x_new)):
			b = self.RG.choice(range(0, a))
			self.swap(x_new, a, b)
		return x_new

	
	def soft_cce(self, y_pred, y_true):
		loss = -torch.sum(y_true*torch.log_softmax(y_pred, dim=1), dim=1)
		return loss.mean()


	def get_mixed_samples(self, X, y, domain_ids, mixup_level):

		batch_ratios = self.RG.beta(self.mixup_beta, self.mixup_beta, size=X.size(0))
		if mixup_level=='feat':
			ratio = np.expand_dims(batch_ratios, axis=1)
		elif mixup_level=='img':
			ratio = np.expand_dims(batch_ratios, axis=(1, 2, 3))
		ratio = torch.from_numpy(ratio).float().cuda()

		doms = list(range(len(torch.unique(domain_ids))))
		bs = X.size(0) // len(doms)
		selected = self.derange(doms)
		
		permuted_across_dom = torch.cat([(torch.randperm(bs) + selected[i] * bs) for i in range(len(doms))])
		permuted_within_dom = torch.cat([(torch.randperm(bs) + i * bs) for i in range(len(doms))])
		
		ratio_within_dom = torch.from_numpy(self.RG.binomial(1, self.mixup_domain, size=X.size(0)))
		mixed_indices = ratio_within_dom*permuted_within_dom + (1. - ratio_within_dom)*permuted_across_dom
		mixed_indices = mixed_indices.long()

		X_mix = ratio*X + (1-ratio)*X[mixed_indices]
		y_a, y_b = y, y[mixed_indices]

		ratio_vec_gt = torch.zeros([X.size()[0], len(self.tr_classes)]).cuda()
		for i in range(X.size()[0]):
			ratio_vec_gt[i, y_a[i]] += batch_ratios[i]
			ratio_vec_gt[i, y_b[i]] += 1-batch_ratios[i]

		return X_mix, y_a, y_b, torch.from_numpy(batch_ratios).float().cuda(), ratio_vec_gt
	
	
	def do_epoch(self):

		self.model.train()

		batch_time = AverageMeter()
		dist_cce_loss = AverageMeter()
		emb_mse_loss = AverageMeter()
		ratio_loss = AverageMeter()
		total_loss = AverageMeter()

		# Start counting time
		time_start = time.time()

		for i, (im, cl, domain_ids) in enumerate(self.train_loader):

			# Transfer im to cuda
			im = im.float().cuda()
			# Get numeric classes
			cls_numeric = torch.from_numpy(utils.numeric_classes(cl, self.dict_clss)).long().cuda()

			self.optimizer.zero_grad()

			if self.args.mixup_level=='img':
				im, y_a, y_b, ratios, ratio_vec_gt = self.get_mixed_samples(im, cls_numeric, domain_ids, 'img')
			
			ratio_vec_pred, features = self.model(im)

			if self.args.mixup_level=='feat':
				features, y_a, y_b, ratios, ratio_vec_gt = self.get_mixed_samples(features, cls_numeric, domain_ids, 'feat')
			
			sem_out = self.model.base_model.last_linear(features)
			
			# Optimize parameters
			cce_l = self.retrieval_loss(sem_out, y_a, y_b, ratios)
			mse_l = self.embedding_loss(sem_out, y_a, y_b, ratios)
			rat_l = self.soft_cce(ratio_vec_pred, ratio_vec_gt)
			loss = self.args.wcce*cce_l + self.args.wmse*mse_l + self.args.wratio*rat_l
			loss.backward()
			
			self.optimizer.step()

			# Store losses for visualization
			dist_cce_loss.update(cce_l.item(), im.size(0))
			emb_mse_loss.update(mse_l.item(), im.size(0))
			ratio_loss.update(rat_l.item(), im.size(0))
			total_loss.update(loss.item(), im.size(0))

			# time
			time_end = time.time()
			batch_time.update(time_end - time_start)
			time_start = time_end

			if (i + 1) % self.args.log_interval == 0:
				print('[Train] Epoch: [{0}/{1}][{2}/{3}]\t'
					  # 'lr:{3:.6f}\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'cce {cce.val:.4f} ({cce.avg:.4f})\t'
					  'mse {mse.val:.4f} ({mse.avg:.4f})\t'
					  'rat {rat.val:.4f} ({rat.avg:.4f})\t'
					  'net {net.val:.4f} ({net.avg:.4f})\t'
					  .format(self.current_epoch+1, self.args.epochs, i+1, len(self.train_loader), batch_time=batch_time, 
							  cce=dist_cce_loss, mse=emb_mse_loss, rat=ratio_loss, net=total_loss))

			# if (i+1)==50:
			#     break

		return {'dist_cce':dist_cce_loss.avg, 'emb_mse':emb_mse_loss.avg, 'ratio_cce':ratio_loss.avg, 'net':total_loss.avg}

	
	def do_training(self):

		print('***Train***')
		for self.current_epoch in range(self.start_epoch, self.args.epochs):

			start = time.time()

			self.adjust_learning_rate()

			self.mixup_beta = min(self.args.mixup_beta, max(self.args.mixup_beta*(self.current_epoch)/self.args.mixup_step, 0.1))
			self.mixup_domain = min(1.0, max((2*self.args.mixup_step - self.current_epoch)/self.args.mixup_step, 0.0))
			print(f'\nAcross Class Mix Coeff:{self.mixup_beta}; Within Domain Mix Coeff:{self.mixup_domain}.\n')
			loss = self.do_epoch()

			# evaluate on validation set, map_ since map is already there
			print('\n***Validation***')
			valid_data = evaluate(self.va_loader_query, self.va_loader_gallery, self.model, self.glove_embed_seen, 
								  self.current_epoch+1, self.args, 'val')
			
			map_ = np.mean(valid_data[self.map_metric])
			prec = valid_data[self.prec_metric]

			end = time.time()
			elapsed = end-start

			print(f"Epoch Time:{elapsed//60:.0f}m{elapsed%60:.0f}s lr:{utils.get_lr(self.optimizer):.7f} mAP:{map_:.4f} prec:{prec:.4f}\n")
			
			if map_ > self.best_map:
				
				self.best_map = map_
				self.early_stop_counter = 0
				
				model_save_name = 'val_map-'+'{0:.4f}'.format(map_)+'_prec-'+'{0:.4f}'.format(prec)+'_ep-'+str(self.current_epoch+1)+self.suffix
				utils.save_checkpoint({
										'epoch':self.current_epoch+1, 
										'model_state_dict':self.model.state_dict(),
										'optimizer_state_dict':self.optimizer.state_dict(), 
										'best_map':self.best_map,
										'corr_prec':prec
										}, directory=self.path_cp, save_name=model_save_name, last_chkpt=self.last_chkpt_name)
				self.last_chkpt_name = model_save_name
			
			else:
				self.early_stop_counter += 1
				if self.args.early_stop==self.early_stop_counter:
					print(f"Validation Performance did not improve for {self.args.early_stop} epochs."
						  f"Early stopping by {self.args.epochs-self.current_epoch-1} epochs.")
					break
				
				print(f"Val mAP hasn't improved from {self.best_map:.4f} for {self.early_stop_counter} epoch(s)!\n")

			# Logger step
			self.logger.add_scalar('Train/glove Based CE loss', loss['dist_cce'], self.current_epoch)
			self.logger.add_scalar('Train/Embedding MSE loss', loss['emb_mse'], self.current_epoch)
			self.logger.add_scalar('Train/Mixup Ratio SoftCCE loss', loss['ratio_cce'], self.current_epoch)
			self.logger.add_scalar('Train/total loss', loss['net'], self.current_epoch)
			self.logger.add_scalar('Val/map', map_, self.current_epoch)
			self.logger.add_scalar('Val/prec', prec, self.current_epoch)

		self.logger.close()

		print('\n***Training and Validation complete***')


def evaluate(loader_sketch, loader_image, model, glove_embed_seen, epoch, args):

	# Switch to test mode
	model.eval()

	batch_time = AverageMeter()

	# Start counting time
	time_start = time.time()

	for i, (sk, cls_sk) in enumerate(loader_sketch):

		sk = sk.float().cuda()

		# Sketch embedding into a semantic space
		with torch.no_grad():
			_, sk_feat = model(sk)
			sk_em = model.base_model.last_linear(sk_feat)

		# Accumulate sketch embedding
		if i == 0:
			acc_sk_em = sk_em.cpu().data.numpy()
			acc_cls_sk = cls_sk
		else:
			acc_sk_em = np.concatenate((acc_sk_em, sk_em.cpu().data.numpy()), axis=0)
			acc_cls_sk = np.concatenate((acc_cls_sk, cls_sk), axis=0)

		# time
		time_end = time.time()
		batch_time.update(time_end - time_start)
		time_start = time_end

		if (i + 1) % args.log_interval == 0:
			print('[Eval][Query] Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  .format(epoch, i + 1, len(loader_sketch), batch_time=batch_time))

	for i, (im, cls_im) in enumerate(loader_image):

		im = im.float().cuda()

		# Image embedding into a semantic space
		with torch.no_grad():
			_, im_feat = model(im)
			im_em = model.base_model.last_linear(im_feat)

		# Accumulate sketch embedding
		if i == 0:
			acc_im_em = im_em.cpu().data.numpy()
			acc_cls_im = cls_im
		else:
			acc_im_em = np.concatenate((acc_im_em, im_em.cpu().data.numpy()), axis=0)
			acc_cls_im = np.concatenate((acc_cls_im, cls_im), axis=0)

		# time
		time_end = time.time()
		batch_time.update(time_end - time_start)
		time_start = time_end

		if (i + 1) % args.log_interval == 0:
			print('[Eval][Gallery] Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  .format(epoch, i + 1, len(loader_image), batch_time=batch_time))
	
	print('\nQuery Emb Dim:{}; Gallery Emb Dim:{}'.format(acc_sk_em.shape, acc_im_em.shape))
	eval_data = compute_retrieval_metrics(acc_sk_em, acc_cls_sk, acc_im_em, acc_cls_im)

	return eval_data