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
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
import torch as t
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

os.chdir("/Users/nuno_chicoria/Documents/master_thesis/files/rsa_ss_nn")

# Class for dataset loading and calling
class Dataset(Dataset):
    
    def __init__(self, filename):
        pd_data = pd.read_csv(filename).values
        self.x = t.FloatTensor(pd_data[:, 0:300])
        self.y_rsa = t.LongTensor(pd_data[:, 300])
        self.y_ss = t.LongTensor(pd_data[:, 301:])
        self.n_samples = self.x.shape[0]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        return self.x[index], self.y_rsa[index], self.y_ss[index]
    
# Neural network class
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.ff = nn.Sequential(nn.Linear(300, 400), nn.ReLU(),
                                nn.Linear(400, 200), nn.ReLU(),
                                nn.Linear(200, 100), nn.ReLU(),
                                nn.Linear(100, 10), nn.ReLU(), t.nn.Dropout(0.1))
        self.rsa = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())
        self.ss = nn.Sequential(nn.Linear(10, 3), nn.Softmax(dim = 1))

    def forward(self, x):
        out = self.ff(x)
        rsa = self.rsa(out)
        ss = self.ss(out)	
        return rsa, ss
    
# MAIN METHOD
window_size = 15
dataset = Dataset(f"dataset_{window_size}.csv")
trainset, testset = train_test_split(dataset)

# =============================================================================
# net = NeuralNetClassifier(
#     Net,
#     max_epochs=20,
#     lr=0.1,
#     optimizer = t.optim.Adam,
#     criterion=nn.BCELoss,
#     iterator_train__shuffle=False,
# )
# 
# net.fit(dataset.x, dataset.y_rsa.float())
# =============================================================================

trainloader = DataLoader(trainset, batch_size = int(len(trainset)/50))
testloader = DataLoader(testset, batch_size = 1)

criterion_rsa = nn.BCELoss(size_average = False)
criterion_ss = nn.BCELoss(size_average = False)

model = Net()
optimizer = t.optim.Adam(model.parameters(), lr = 1e-2, weight_decay = 1)

for epoch in range(10):
    for sample in trainloader:
        x, yrsa, yss = sample
        optimizer.zero_grad()
        yprsa, ypss = model(Variable(x))
        loss_rsa = criterion_rsa(yprsa, Variable(yrsa).float())
        loss_ss = criterion_ss(ypss, Variable(yss).float())
        (loss_rsa + loss_ss).backward()
        optimizer.step()

t.save(model, "rsa_ss_nn.pt")
        
