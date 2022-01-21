import numpy as np
import torch

import logging

from cifar10ds import load_cifar10_data

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    # logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts

'''
datair: where the dataset is located
n_nets: how many datasets to generate; the number of clients in FL
alpha: parameter of dirichlet distribution, larger alpha means more strong non iid condition
K: classes of the dataset
selection: use a repeat non iid distribution, used for debugging codes
'''
def partition_data(datadir,n_nets, alpha=0.5,K=10,selection=None):
    # functions to load data, here I use a cifar10 as example
    X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    n_train = X_train.shape[0]
    min_size = 0
    K = 10
    N = y_train.shape[0]
    net_dataidx_map = {}
    if selection==None:
        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                '''
                making every clients have the same dataset size, not necessary for all FL environment
                '''
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
    else:
        idx_batch = [[] for _ in range(n_nets)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            point=0
            for j in range(n_nets):
                idx_batch[j].extend(idx_k[point:point+selection[j][k]])
                point+=selection[j][k]
        
    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

if __name__=='__main__':
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data('./data',10)
    
