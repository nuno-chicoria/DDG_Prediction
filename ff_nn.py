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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

os.chdir("/Users/nuno_chicoria/Documents/master_thesis/files/ff_nn")

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
    
class LossWrapperCE():
	
	def __init__(self, loss,  dummyColumn=False):
		self.loss = loss
		self.dummyColumn = dummyColumn
		
	def __call__(self, input, target):
		if self.dummyColumn == False:
			input = t.cat([1-input, input],1)
		else:
			tmp = Variable(t.ones(input.size()[0], input.size()[1]))
			if input.is_cuda:
				tmp = tmp.cuda()		
			input = t.cat([tmp, input],1)
		return self.loss(input, target)
    
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.ff = nn.Sequential(nn.Linear(220, 25), nn.ReLU(), nn.Linear(25, 10), nn.ReLU(), t.nn.Dropout(0.1))
        self.rsa = nn.Sequential(nn.Linear(10, 1))
        self.ss = nn.Sequential(nn.Linear(10, 3), nn.Softmax(dim = 0))

    def forward(self, x):
        out = self.ff(x)
        rsa = self.rsa(out)
        ss = self.ss(out)	
        return rsa, ss
    
# =============================================================================

# =============================================================================
# dataset = Dataset("dataset.csv")
# trainset, testset = train_test_split(dataset)
# =============================================================================

trainloader = DataLoader(trainset, batch_size = int(len(trainset)/100))
testloader = DataLoader(testset)

criterion_rsa = LossWrapperCE(nn.CrossEntropyLoss(size_average=False), True)
criterion_ss = nn.BCELoss(size_average=False)

model = Net()
optimizer = t.optim.Adam(model.parameters(), lr = 1e-2, weight_decay = 1)

for epoch in tqdm(range(20)):
    for sample in trainloader:
        x, yrsa, yss = sample
        optimizer.zero_grad()
        yprsa, ypss = model(Variable(x))
        loss_rsa = criterion_rsa(yprsa, Variable(yrsa))
        loss_ss = criterion_ss(ypss, Variable(yss).float())
        (loss_rsa + loss_ss).backward()
        optimizer.step()

rsa_true = []
rsa_pred = []
ss_true = []
ss_pred = []
        
for sample in testloader:
    x, yrsa, yss = sample
    rsa_true.append(yrsa.numpy())
    ss_true.append(yss.numpy())
    yprsa, ypss = model(Variable(x))
    if yprsa > 0:
        rsa_pred.append([1])
    else:
        rsa_pred.append([0])
    if ypss[0][0] > ypss[0][1] and ypss[0][0] > ypss[0][2]:
        ss_pred.append([1, 0, 0])
    elif ypss[0][1] > ypss[0][0] and ypss[0][1] > ypss[0][2]:
        ss_pred.append([0, 1, 0])
    else:
        ss_pred.append([0, 0, 1])

rsa_conf = confusion_matrix(rsa_true, rsa_pred)
ss_conf = confusion_matrix(ss_true, ss_pred)

