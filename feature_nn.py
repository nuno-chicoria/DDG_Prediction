#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for a Feed Forward Neural Network for the prediction of RSA and SS using
frequency table of MSAs as features and RSA and SS as labels.

3. Neural Network Building and Training

@author: nuno_chicoria
"""

import numpy as np
import os
import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import KFold
import torch as t
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

os.chdir("/Users/nuno_chicoria/Documents/master_thesis/files/rsa_ss_nn")

# Class for dataset loading and calling
class Dataset(Dataset):
    
    def __init__(self, filename):
        pd_data = pd.read_csv(filename).values
        self.x = t.FloatTensor(pd_data[:, 0:300])
        self.y_rsa = t.FloatTensor(pd_data[:, 300])
        self.y_ss = t.FloatTensor(pd_data[:, 301:])
        self.n_samples = self.x.shape[0]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        return self.x[index], self.y_rsa[index], self.y_ss[index]
    
# Neural network class
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.ff = nn.Sequential(nn.Linear(300, 100), nn.ReLU(),
                                nn.Linear(100, 10), nn.ReLU(), t.nn.Dropout(0.1))
        self.rsa = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())
        self.ss = nn.Sequential(nn.Linear(10, 3), nn.Softmax(dim = 0))

    def forward(self, x):
        out = self.ff(x)
        rsa = self.rsa(out)
        ss = self.ss(out)	
        return rsa, ss
    
# MAIN METHOD
window_size = 15
dataset = Dataset(f"dataset_{window_size}.csv")
<<<<<<< HEAD
cv = KFold(n_splits = 10, random_state = 42, shuffle=False)
scores = []

for train_index, test_index in cv.split(dataset):
    
    trainset, testset = dataset[train_index], dataset[test_index]
    trainloader = DataLoader(trainset, batch_size = int(len(trainset[0])/50))
    testloader = DataLoader(testset, batch_size = 1)
    
    criterion_rsa = nn.BCELoss(size_average=False)
    criterion_ss = nn.BCELoss(size_average=False)
    
    model = Net()
    optimizer = t.optim.Adam(model.parameters(), lr = 1e-2, weight_decay = 1)
    
    for epoch in range(20):
        print(f"{epoch+1}/20")
        for sample in trainloader:
            x, yrsa, yss = sample
            optimizer.zero_grad()
            yprsa, ypss = model(Variable(x))
            loss_rsa = criterion_rsa(yprsa, Variable(yrsa).float())
            loss_ss = criterion_ss(ypss, Variable(yss).float())
            (loss_rsa + loss_ss).backward()
            optimizer.step()
    
    rsa_true = []
    rsa_pred = []
    ss_true = []
    ss_pred = []
        
    for sample in testloader:
=======
trainset, testset = train_test_split(dataset)

trainloader = DataLoader(trainset, batch_size = int(len(trainset)/50))

criterion_rsa = nn.BCELoss(size_average=False)
criterion_ss = nn.BCELoss(size_average=False)

model = Net()
optimizer = t.optim.Adam(model.parameters(), lr = 1e-2, weight_decay = 1)

for epoch in tqdm(range(20)):
    for sample in trainloader:
>>>>>>> parent of cfad5dc... 01.06.2019 (SoftMax & DataLoader Debugging)
        x, yrsa, yss = sample
        yprsa, ypss = model(Variable(x))
        rsa_true.append(yrsa[0].numpy())
        ss_true.append(yss[0].numpy())
        rsa_pred.append(yprsa[0].detach().numpy())
        ss_pred.append(ypss[0].detach().numpy())
    
    fpr, tpr, thresholds = metrics.roc_curve(rsa_true, rsa_pred)
    optimal_idx = np.argmax(np.abs(tpr - fpr))
    rsa_threshold = thresholds[optimal_idx]
    
    rsa_binary = []
    for i in range(len(rsa_pred)):
        if rsa_pred[i] >= rsa_threshold:
            rsa_binary.append(1)
        else:
            rsa_binary.append(0)
    scores.append(metrics.accuracy_score(rsa_true, rsa_binary))
    print(scores)

t.save(model, "rsa_ss_nn.pt")
        
