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
    
class FF_NN(nn.Module):
    
    def __init__(self):
        super(FF_NN, self).__init__()
        self.ff = nn.Sequential(nn.Linear(220, 25), nn.ReLU(), nn.Linear(25, 10), nn.ReLU())
        self.rsa = nn.Sequential(nn.Linear(10, 1))
        self.ss = nn.Sequential(nn.Linear(10, 3), nn.Softmax(dim = 0))

    def forward(self, x):
        out = self.ff(x)
        rsa = self.rsa(out)
        ss = self.ss(out)	
        return rsa, ss
    
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

# =============================================================================

#dataset = Dataset("dataset.csv")

dataloader = DataLoader(dataset, batch_size = int(len(dataset)/100), shuffle = True, sampler = None, num_workers = 0)

model = FF_NN()
criterion_rsa = LossWrapperCE(nn.CrossEntropyLoss(size_average=False), True)
criterion_ss = nn.BCELoss(size_average=False)
optimizer = t.optim.Adam(model.parameters(), lr = 1e-2, weight_decay = 1)

for epoch in tqdm(range(5)):
    for sample in dataloader:
        x, y_rsa, y_ss = sample
        optimizer.zero_grad()
        yp_rsa, yp_ss = model(Variable(x))
        loss_rsa = criterion_rsa(yp_rsa, Variable(y_rsa))
        loss_ss = criterion_ss(yp_ss, Variable(y_ss).float())
        (loss_rsa + loss_ss).backward()
        optimizer.step()
        
