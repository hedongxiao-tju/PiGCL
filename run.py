#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv

from model import Encoder, Model, drop_feature
from eval import label_classification, cluster_test

from Dataset_Load import load_dataset
from warnings import filterwarnings

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# In[2]:


def semi_grad_function(z1, z2, neg_matrix, ex_r, args):
    f = lambda x: torch.exp(x / args.tau)
    sim_t = F.normalize(z1) @ F.normalize(z2).T
    sim1 = f(sim_t)
    sim2 = f(F.normalize(z1) @ F.normalize(z1).T)
    sim3 = f(F.normalize(z2) @ F.normalize(z2).T)
    #print(sim1.diag())
    l1 = -torch.log(sim1.diag() / (((sim1*neg_matrix).sum(dim=1)+(sim2*neg_matrix).sum(dim=1)) * ex_r +sim1.diag() ) )
    l2 = -torch.log(sim1.diag() / (((sim1*neg_matrix).sum(dim=1)+(sim3*neg_matrix).sum(dim=1)) * ex_r +sim1.diag() ) )

    print(((l1 + l2)/2).mean())
    print(sim_t.diag().mean() * (1-1/ex_r))
    
    return ((l1 + l2)/2).mean() - sim_t.diag().mean() * (1-1/ex_r)
    


# In[3]:


def train(model: Model, x, edge_index, optimizer, neg_matrix, ex_r, args):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=args.drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=args.drop_edge_rate_2)[0]
    x_1 = drop_feature(x, args.drop_feature_rate_1)
    x_2 = drop_feature(x, args.drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = semi_grad_function(model.projection(z1), model.projection(z2), neg_matrix, ex_r, args)
    
    loss.backward()
    optimizer.step()

    return loss.item()

def get_sim(model: Model, x, edge_index):
    with torch.no_grad():
        model.eval()
        z = model(x, edge_index)
        return F.normalize(z) @ F.normalize(z).T
        
        

def print_class_sim(model: Model, x, edge_index, y, if_print=True, if_write=False):
    model.eval()
    z = model(x, edge_index)
    l_n = torch.unique(y).size()[0]
    mean_ = torch.zeros([l_n,l_n])
    std_ = torch.zeros([l_n,l_n])
    for i in range(l_n):
        z0 = z[y==i]
        for j in range(l_n):
            z1 = z[y==j]
            c = F.normalize(z0) @ F.normalize(z1).T
            mean_[i,j] = c.mean()
            std_[i,j] = c.std()
    
    if if_print:
        print('mean:')
        print(mean_)
        print('std:')
        print(std_)
    
    if if_write:
        with open(args.log_dir, 'a') as f:
            f.write(str(mean_) + '\n')
        #return r

    return mean_, std_
            


def test(model: Model, x, edge_index, y, args, epoch_num=-1, final=False):
    model.eval()
    z = model(x, edge_index)
    assert args.test_kind in ['classification', 'clustering']
    if args.test_kind == 'classification':
        r = label_classification(z, y, ratio=0.1)
    elif args.test_kind == 'clustering':
        r = cluster_test(z.detach().cpu(), torch.unique(y).size()[0], y.detach().cpu(), args.cluster_random_state)
    with open(args.log_dir, 'a') as f:
        f.write(str(epoch_num) + ':\t')
        f.write(str(r) + '\n')
    
    return r


# In[4]:


def run(args):

    with open(args.log_dir, 'a') as f:
        f.write('\n\n'+'##'*20+'\n')
        f.write(str(args) + '\n')
        
    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    torch.manual_seed(args.seed)
    random.seed(12345)

    learning_rate = args.learning_rate
    num_hidden = args.num_hidden
    num_proj_hidden = args.num_proj_hidden
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[args.activation]
    base_model = ({'GCNConv': GCNConv})[args.base_model]
    num_layers = args.num_layers

    drop_edge_rate_1 = args.drop_edge_rate_1
    drop_edge_rate_2 = args.drop_edge_rate_2
    drop_feature_rate_1 = args.drop_feature_rate_1
    drop_feature_rate_2 = args.drop_feature_rate_2
    tau = args.tau
    num_epochs = args.num_epochs
    weight_decay = args.weight_decay

    #path = './datasets/'
    dataset = load_dataset(args.dataset_name, args.dataset_dir)
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    encoder = Encoder(dataset.num_features, num_hidden, activation,
                      base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    neg_matrix = torch.ones(data.x.size()[0],data.x.size()[0]).to(device)
    
    start = t()
    prev = start
    ex_r = 1
    for epoch in range(1, num_epochs + 1):
        if epoch == 1:
            loss = train(model, data.x, data.edge_index, optimizer, neg_matrix, ex_r, args)
            ori_sim = get_sim(model, data.x, data.edge_index)
        else:
            loss = train(model, data.x, data.edge_index, optimizer, neg_matrix, ex_r, args)
        
        now = t()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
              f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now
        
        if epoch % args.update_negmatrix_epoch == 0:
            with torch.no_grad():
                sim = get_sim(model, data.x, data.edge_index)
                sim_grad = sim - ori_sim
                te = sim_grad.reshape(-1).sort()[0][int(sim_grad.size()[0]*sim_grad.size()[1]*args.non_rate)]
                neg_matrix = torch.zeros_like(sim_grad).to(device)
                neg_matrix[sim_grad>=te] = 1
                ex_r = sim.mean() / ((sim * neg_matrix).mean() )
                ori_sim = sim
            
        
        if epoch % args.num_epochs_test == 0:
            test(model, data.x, data.edge_index, data.y, args, epoch_num=epoch)

    print("=== Final ===")
    print_class_sim(model, data.x, data.edge_index, data.y, if_print=True, if_write=True)
    test(model, data.x, data.edge_index, data.y, args, final=False)



# In[5]:


if __name__ == '__main__':
    filterwarnings(action='ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Cora')
    parser.add_argument('--dataset_dir', type=str, default='./datasets/')
    parser.add_argument('--gpu_id', type=int, default=0)
    
    parser.add_argument('--seed', type=int, default=39788)
    parser.add_argument('--learning_rate', type=float, default=0.0005) 
    parser.add_argument('--num_hidden', type=int, default=128)
    parser.add_argument('--num_proj_hidden', type=int, default=128)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--base_model', type=str, default='GCNConv')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--drop_edge_rate_1', type=float, default=0.2)
    parser.add_argument('--drop_edge_rate_2', type=float, default=0.4)
    parser.add_argument('--drop_feature_rate_1', type=float, default=0.3)
    parser.add_argument('--drop_feature_rate_2', type=float, default=0.4)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--num_epochs_test', type=int, default=50)
    parser.add_argument('--update_negmatrix_epoch', type=int, default=20)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--non_rate', type=float, default=0.05)
    parser.add_argument('--test_kind', type=str, default='classification') # classification, clustering
    parser.add_argument('--cluster_random_state', type=int, default='12345')
    parser.add_argument('--log_dir', type=str, default='./log/log_Cora.txt')
    args = parser.parse_args()
    
    with open(args.log_dir, 'a') as f:
        f.write('\n\n'+'****'*20+'\n')
        f.write('****'*20+'\n')
        f.write('****'*20+'\n')                
    run(args)
            

