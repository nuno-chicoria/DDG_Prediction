#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:01:29 2019

@author: nuno_chicoria
"""

import glob
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
 
os.chdir("/Users/nuno_chicoria/Documents/master_thesis/files/ff_nn")

class Dataset(Dataset):
    
    def __init__(self, filename):
        pd_data = pd.read_csv(filename).values
        self.data = pd_data[:, 0:200]
        self.target = pd_data[:, 220:]
        self.n_samples = self.data.shape[0]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        return torch.Tensor(self.data[index]), torch.Tensor(self.target[index])

class FeedforwardNeuralNetModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.ff = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, output_dim), nn.Sigmoid())

    def forward(self, x):
        out = self.ff(x)
        return out

# =============================================================================

name = str
batch_size = 100
input_dim = 220
hidden_dim = 20
output_dim = 1
learning_rate = 0.1

for filepath in glob.iglob("/Users/nuno_chicoria/Documents/master_thesis/files/msa_tensor/*.pt"):
    name = os.path.basename(filepath).partition(".")[0]
    msa = torch.load(filepath)
    rsa_torch = torch.load("/Users/nuno_chicoria/Documents/master_thesis/files/rsa_tensor/%s.pt" % name).type(torch.LongTensor)
    ss_torch = torch.load("/Users/nuno_chicoria/Documents/master_thesis/files/ss_tensor/%s.pt" % name).type(torch.LongTensor)
    msa = np.concatenate((msa, rsa_torch), axis=1)
    msa = np.concatenate((msa, ss_torch), axis=1)
    with open("dataset.csv", "a") as file:
        np.savetxt(file, msa, delimiter = ",")

print("Loading dataset")
my_data = Dataset("dataset.csv")

print("Preparing dataloader")
dataloader = DataLoader(my_data, batch_size = batch_size, shuffle = True, num_workers = 0)

print("Initialising neural network")
model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim).double()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-4)

for epoch in range(5):
    print("Current epoch: %s" % str(epoch + 1))
    for i, (features, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
