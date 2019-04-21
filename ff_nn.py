#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:01:29 2019

@author: nuno_chicoria
"""

import os
import pandas as pd
import torch as t
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

 
os.chdir("/Users/nuno_chicoria/Documents/master_thesis/files/ff_nn")

class Dataset(Dataset):
    
    def __init__(self, filename):
        pd_data = pd.read_csv(filename).values
        self.x = t.Tensor(pd_data[:, 0:220])
        self.y_rsa = t.Tensor(pd_data[:, 220])
        self.y_ss = t.Tensor(pd_data[:, 221:])
        self.n_samples = self.x.shape[0]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        return self.x[index], self.y_rsa[index], self.y_ss[index]
    
class FF_NN(nn.Module):
    
    def __init__(self):
        super(FF_NN, self).__init__()
        self.ff = nn.Sequential(nn.Linear(220, 100), nn.ReLU(), nn.Linear(100, 50), nn.ReLU())
        self.rsa = nn.Sequential(nn.Linear(50, 1), nn.Sigmoid())
        self.ss = nn.Sequential(nn.Linear(50, 3), nn.Softmax())

    def forward(self, x):
        out = self.ff(x)
        rsa = self.rsa(out)
        ss = self.ss(out)	
        return rsa, ss

# =============================================================================

print("Loading dataset")
dataset = Dataset("dataset.csv")

print("Preparing dataloader")
dataloader = DataLoader(dataset, batch_size = int(len(dataset)/100), shuffle = True, sampler = None, num_workers = 0)

print("Initialising neural network")
model = FF_NN()
criterion_rsa = nn.BCELoss()
criterion_ss = nn.MSELoss()
optimizer = t.optim.Adam(model.parameters(), lr = 1e-2, weight_decay = 1)

for epoch in range(100):
    print("Epoch %s" % str(epoch + 1))
    for sample in dataloader:
        x, y_rsa, y_ss = sample
        optimizer.zero_grad()
        yprsa, ypss = model(Variable(x))
        loss_rsa = criterion_rsa(yprsa, Variable(y_rsa))
        loss_ss = criterion_ss(ypss, Variable(y_ss))
        (loss_rsa + loss_ss).backward()
        optimizer.step()
        
