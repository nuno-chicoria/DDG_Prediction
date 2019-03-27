#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for a Feed Forward Neural Network for the prediction of RSA and SS using
frequency table of MSAs as features and RSA and SS as labels.

3. Feed Forward Neural Network

@author: nuno_chicoria
"""

import glob
import os
from sklearn.model_selection import KFold
import torch
import torch.nn as nn


#Import file names
msa_files = []
for filepath in glob.iglob("/Users/nuno_chicoria/Documents/master_thesis/msa_tensors/*.pt"):
    name = os.path.basename(filepath).partition(".")[0]
    msa_files.append(name)
    
#Indices for the k-fold cross validation
kf = KFold(n_splits = 5)

for train_index, test_index in kf.split(msa_files):
    names_train, names_test = msa_files[train_index], msa_files[test_index]

# =============================================================================
# os.chdir("/Users/nuno_chicoria/Documents/master_thesis/ff_nn")
# 
# #Import data
# X = torch.tensor(([2, 9], [1, 5], [3, 6]), dtype=torch.float)
# y = torch.tensor(([92], [100], [89]), dtype=torch.float)
# 
# #Building the Neural Network
# #Class header indicates that we are defininf a customized nn "nn.Module"
# class Neural_Network(nn.Module):
#     
#     #Initialization
#     #
#     def __init__(self, ):
#         super(Neural_Network, self).__init__()
#         # parameters
#         self.inputSize = 2
#         self.outputSize = 1
#         self.hiddenSize = 3
#         
#         # weights
#         self.W1 = torch.randn(self.inputSize, self.hiddenSize) # 3 X 2 tensor
#         self.W2 = torch.randn(self.hiddenSize, self.outputSize) # 3 X 1 tensor
#         
#     def forward(self, X):
#         self.z = torch.matmul(X, self.W1) # 3 X 3 ".dot" does not broadcast in PyTorch
#         self.z2 = self.sigmoid(self.z) # activation function
#         self.z3 = torch.matmul(self.z2, self.W2)
#         o = self.sigmoid(self.z3) # final activation function
#         return o
#         
#     def sigmoid(self, s):
#         return 1 / (1 + torch.exp(-s))
#     
#     def sigmoidPrime(self, s):
#         # derivative of sigmoid
#         return s * (1 - s)
#     
#     def backward(self, X, y, o):
#         self.o_error = y - o # error in output
#         self.o_delta = self.o_error * self.sigmoidPrime(o) # derivative of sig to error
#         self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
#         self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
#         self.W1 += torch.matmul(torch.t(X), self.z2_delta)
#         self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)
#         
#     def train(self, X, y):
#         # forward + backward pass for training
#         o = self.forward(X)
#         self.backward(X, y, o)
#         
#     def saveWeights(self, model):
#         # we will use the PyTorch internal storage functions
#         torch.save(model, "NN")
#         # you can reload model with all the weights and so forth with:
#         # torch.load("NN")
# =============================================================================
        
