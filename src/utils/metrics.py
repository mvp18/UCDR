import sys
import time
import numpy as np

import multiprocessing
from joblib import Parallel, delayed

# metrics
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score


def prec(actual, predicted, k):
    
    act_set = set(actual)
    
    if k is not None:
        pred_set = set(predicted[:k])
        pr = len(act_set & pred_set) / min(k, len(pred_set))
    else:
        pred_set = set(predicted)
        pr = len(act_set & pred_set) / max(len(pred_set), 1)
    
    return pr


def rec(actual, predicted, k):
    
    act_set = set(actual)
    pred_set = set(predicted[:k])
    re = len(act_set & pred_set) / max(len(act_set), 1)
    
    return re


def precak(sim, str_sim, k=None):
    
    act_lists = [np.nonzero(s)[0] for s in str_sim]
    pred_lists = np.argsort(-sim, axis=1)
    # num_cores = min(multiprocessing.cpu_count(), 2)
    num_cores = multiprocessing.cpu_count()
    nq = len(act_lists)
    preck = Parallel(n_jobs=num_cores)(delayed(prec)(act_lists[iq], pred_lists[iq], k) for iq in range(nq))
    reck = Parallel(n_jobs=num_cores)(delayed(rec)(act_lists[iq], pred_lists[iq], k) for iq in range(nq))
    # preck = [prec(act_lists[iq], pred_lists[iq], k) for iq in range(nq)]
    # reck = [rec(act_lists[iq], pred_lists[iq], k) for iq in range(nq)]
    
    return np.mean(preck), np.mean(reck)


def aps(sim, str_sim):
    nq = str_sim.shape[0]
    # num_cores = min(multiprocessing.cpu_count(), 2)
    num_cores = multiprocessing.cpu_count()
    aps = Parallel(n_jobs=num_cores)(delayed(average_precision_score)(str_sim[iq], sim[iq]) for iq in range(nq))
    # aps = [average_precision_score(str_sim[iq], sim[iq]) for iq in range(nq)]
    return aps


def apsak(sim, str_sim, k=None):
    idx = (-sim).argsort()[:, :k]
    sim_k = np.array([sim[i, id] for i, id in enumerate(idx)])
    str_sim_k = np.array([str_sim[i, id] for i, id in enumerate(idx)])
    idx_nz = np.where(str_sim_k.sum(axis=1) != 0)[0]
    sim_k = sim_k[idx_nz]
    str_sim_k = str_sim_k[idx_nz]
    aps_ = np.zeros((sim.shape[0]), dtype=np.float)
    aps_[idx_nz] = aps(sim_k, str_sim_k)
    return aps_


def compute_retrieval_metrics(acc_sk_em, acc_cls_sk, acc_im_em, acc_cls_im):

    print('\nComputing evaluation metrics...', end='')

    # Compute similarity
    t = time.time()
    sim_euc = np.exp(-cdist(acc_sk_em, acc_im_em, metric='euclidean'))
    time_euc = (time.time() - t) / acc_cls_sk.shape[0]

    # similarity of classes or ground truths
    # Multiplied by 1 for boolean to integer conversion
    str_sim = (np.expand_dims(acc_cls_sk, axis=1) == np.expand_dims(acc_cls_im, axis=0))*1

    apsall = apsak(sim_euc, str_sim)
    aps200 = apsak(sim_euc, str_sim, k=200)
    
    prec100, _ = precak(sim_euc, str_sim, k=100)
    prec200, _ = precak(sim_euc, str_sim, k=200)

    eval_data = {'aps@all': apsall, 'aps@200': aps200, 'prec@200': prec200, 'prec@100': prec100}

    print('Done')

    return eval_data