#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for a Feed Forward Neural Network for the prediction of RSA and SS using
frequency table of MSAs as features and RSA and SS as labels.

3. Neural Network and Performance Evaluation

@author: nuno_chicoria
"""

import os
import pandas as pd
import torch as t
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_curve, classification_report, auc, matthews_corrcoef
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import cycle

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
        self.ff = nn.Sequential(nn.Linear(220, 25), nn.ReLU(),
                                nn.Linear(25, 10), nn.ReLU(), t.nn.Dropout(0.1))
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
testloader = DataLoader(testset, batch_size = 1)

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
        
for sample in testloader:
    x, yrsa, yss = sample
    yprsa, ypss = model(Variable(x))
    break
    rsa_true.append(yrsa.numpy())
    ss_true.append(yss.numpy())
    rsa_pred.append(yprsa.detach().numpy())
    ss_pred.append(ypss.detach().numpy())
    
#RSA ROC Curve
fpr, tpr, thresholds = roc_curve(rsa_true, rsa_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RSA ROC Curve')
plt.legend(loc="lower right")
plt.show()

#SS ROC Curve
ss_true_1 = []
for i in range(len(ss_true)):
    ss_true_1.append(ss_true[i][0])
ss_true_2 = []
for i in range(len(ss_true)):
    ss_true_2.append(ss_true[i][1])
ss_true_3 = []
for i in range(len(ss_true)):
    ss_true_3.append(ss_true[i][2])
ss_pred_1 = []
for i in range(len(ss_pred)):
    ss_pred_1.append(ss_pred[i][0])
ss_pred_2 = []
for i in range(len(ss_pred)):
    ss_pred_2.append(ss_pred[i][1])
ss_pred_3 = []
for i in range(len(ss_pred)):
    ss_pred_3.append(ss_pred[i][2])

fpr1, tpr1, thresholds1 = roc_curve(ss_true_1, ss_pred_1)
fpr2, tpr2, thresholds2 = roc_curve(ss_true_2, ss_pred_2)
fpr3, tpr3, thresholds3 = roc_curve(ss_true_3, ss_pred_3)

roc_auc1 = auc(fpr1, tpr1)
roc_auc2 = auc(fpr2, tpr2)
roc_auc3 = auc(fpr3, tpr3)

fpr = [fpr1, fpr2, fpr3]
tpr = [tpr1, tpr2, tpr3]
roc_auc = [roc_auc1, roc_auc2, roc_auc3]

plt.figure()
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SS ROC Curve')
plt.legend(loc="lower right")
plt.show()
        
#rsa_conf = confusion_matrix(rsa_true, rsa_pred)
#rsa_report = classification_report(y_true = rsa_true, y_pred = rsa_pred)

#ss_matthews = matthews_corrcoef(ss_true, ss_pred)
#ss_conf = confusion_matrix(ss_true, ss_pred)
#ss_report = classification_report(y_true = ss_true, y_pred = ss_pred)



