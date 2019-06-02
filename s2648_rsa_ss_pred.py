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

#model = t.load("/Users/nuno_chicoria/Documents/master_thesis/files/rsa_ss_nn/rsa_ss_nn.pt")

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