#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for the prediction of RSA and SS values for the unique entries of S2648 dataset

@author: nuno_chicoria
"""

import glob
import numpy as np
import os
import torch as t
from torch.autograd import Variable

# Predicting RSA and SS for each entry of the S2648 dataset
name = []
rsa_pred = []
ss_pred = [] 
for filepath in glob.iglob("/Users/nuno_chicoria/Documents/master_thesis/files/s2648_msa/*.npy"):
    msa = np.load(filepath)
    msa = t.from_numpy(msa).float()
    yprsa, ypss = model(Variable(msa))
    name.append(os.path.basename(filepath).partition(".")[0])
    rsa_pred.append(yprsa.detach().numpy())
    ss_pred.append(ypss.detach().numpy())
    
# Converting RSA to binary
rsa_binary = []
for entry in rsa_pred:
    temp = ""
    for i in entry:
        if i >= rsa_threshold:
            temp += "e"
        else:
            temp += "-"
    rsa_binary.append(temp)

# Saving RSA information
file = open("/Users/nuno_chicoria/Documents/master_thesis/files/s2648_rsa_ss/s2648_rsa.txt","w+")
for i in range(len(name)):
     file.write(">" + name[i][0:4] + ":" + name[i][4] + "\n")
     file.write(rsa_binary[i] + "\n")
     
# Converting SS to binary (not working)  
ss_binary = []
ss = ["C", "E", "H"]
for entry in ss_pred:
    temp = ""
    for trio in entry:
        index_max = np.argmax(trio)
        temp += ss[index_max]
    ss_binary.append(temp)

# Saving SS information
file = open("/Users/nuno_chicoria/Documents/master_thesis/files/s2648_rsa_ss/s2648_ss.txt","w+")
for i in range(len(name)):
     file.write(">" + name[i][0:4] + ":" + name[i][4] + "\n")
     file.write(ss_binary[i] + "\n")