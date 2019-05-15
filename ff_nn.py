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
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report, auc, matthews_corrcoef
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        self.ff = nn.Sequential(nn.Linear(220, 25), nn.ReLU(), nn.Linear(25, 10), nn.ReLU(), t.nn.Dropout(0.1))
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

trainloader = DataLoader(trainset, batch_size = int(len(trainset)/100))
#testloader = DataLoader(testset, batch_size = 1)

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

rsa_true = []
rsa_pred = []
ss_true = []
ss_pred = []
        
for i in range(len(testset)):
    x, yrsa, yss = testset[i]
    rsa_true.append(yrsa.numpy())
    ss_true.append(yss.numpy())
    yprsa, ypss = model(Variable(x))
    rsa_pred.append(yprsa.detach().numpy())
    ss_pred.append(ypss.detach().numpy())
    
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(rsa_true[i], rsa_pred[i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

#Precision is the ratio of correctly predicted positive observations to the total predicted positive observations
#Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes
#F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account

#fpr, tpr, thresholds = roc_curve(rsa_true, rsa_pred)
        
#rsa_conf = confusion_matrix(rsa_true, rsa_pred)
#rsa_report = classification_report(y_true = rsa_true, y_pred = rsa_pred)

#ss_conf = confusion_matrix(ss_true, ss_pred)
#ss_report = classification_report(y_true = ss_true, y_pred = ss_pred)

