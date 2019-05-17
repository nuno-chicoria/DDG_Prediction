#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for a Feed Forward Neural Network for the prediction of RSA and SS using
frequency table of MSAs as features and RSA and SS as labels.

3. Neural Network Building and Training

@author: nuno_chicoria
"""

import os
import pandas as pd
import torch as t
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

os.chdir("/Users/nuno_chicoria/Documents/master_thesis/files/ff_nn")

#class for dataset loading and calling
class Dataset(Dataset):
    
    def __init__(self, filename):
        pd_data = pd.read_csv(filename).values
        self.x = t.FloatTensor(pd_data[:, 0:220])
        self.y_rsa = t.LongTensor(pd_data[:, 220])
        self.y_ss = t.LongTensor(pd_data[:, 221:])
        self.n_samples = self.x.shape[0]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        return self.x[index], self.y_rsa[index], self.y_ss[index]
    
#neural network class
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.ff = nn.Sequential(nn.Linear(220, 100), nn.ReLU(),
                                nn.Linear(100, 10), nn.ReLU(), t.nn.Dropout(0.1))
        self.rsa = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())
        self.ss = nn.Sequential(nn.Linear(10, 3), nn.Softmax(dim = 0))

    def forward(self, x):
        out = self.ff(x)
        rsa = self.rsa(out)
        ss = self.ss(out)	
        return rsa, ss
    
#MAIN METHOD
dataset = Dataset("dataset.csv")
trainset, testset = train_test_split(dataset)

trainloader = DataLoader(trainset, batch_size = int(len(trainset)/50))

criterion_rsa = nn.BCELoss(size_average=False)
criterion_ss = nn.BCELoss(size_average=False)

model = Net()
optimizer = t.optim.Adam(model.parameters(), lr = 1e-2, weight_decay = 1)

for epoch in tqdm(range(20)):
    for sample in trainloader:
        x, yrsa, yss = sample
        optimizer.zero_grad()
        yprsa, ypss = model(Variable(x))
        loss_rsa = criterion_rsa(yprsa, Variable(yrsa).float())
        loss_ss = criterion_ss(ypss, Variable(yss).float())
        (loss_rsa + loss_ss).backward()
        optimizer.step()

#save trainset, testset, model
        
