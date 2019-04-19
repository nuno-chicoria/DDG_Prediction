#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:01:29 2019

@author: nuno_chicoria
"""

import glob
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
 
class Dataset(Dataset):
    
    def __init__(self, X, y):
        self.data = X
        self.target = y
        self.n_samples = len(self.data)
    
    def __len__(self):   # Length of the dataset.
        return self.n_samples
    
    def __getitem__(self, index):   # Function that returns one point and one label.
        return self.data[index], self.target[index]

class FeedforwardNeuralNetModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu.ReLU(out)
        out = self.fc2(out)
        return out

#######################
##### MAIN METHOD #####
#######################

batch_size = 1
input_dim = 220
hidden_dim = 20
output_dim = 1
learning_rate = 0.1

file_names = []
dataset = []
rsa = []
ss = []

for filepath in glob.iglob("/Users/nuno_chicoria/Documents/master_thesis/files/msa_tensor/*.pt"):
    name = os.path.basename(filepath).partition(".")[0]
    file_names.append(name)
    temp = torch.load(filepath)
    dataset.append(temp)

for name in file_names:
    temp = torch.load("/Users/nuno_chicoria/Documents/master_thesis/files/rsa_tensor/%s.pt" % name)
    rsa.append(temp)

for name in file_names:
    temp = torch.load("/Users/nuno_chicoria/Documents/master_thesis/files/ss_tensor/%s.pt" % name)
    ss.append(temp)

my_data = Dataset(dataset, ss)
dataloader = DataLoader(my_data, batch_size = batch_size, shuffle = False, num_workers = 0)
model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-4)

for epoch in range(5):
    for i, (features, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()