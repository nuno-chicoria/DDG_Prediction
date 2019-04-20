#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Code for a Feed Forward Neural Network for the prediction of RSA and SS using
frequency table of MSAs as features and RSA and SS as labels.

2. Label Matrix Generation

@author: nuno_chicoria
"""

import os
import numpy as np
import torch

#For loop for the matrices regarding relative solvent accessibility
rsa = open("/Users/nuno_chicoria/Documents/master_thesis/datasets/rsa_ss/ACCpro.dssp.txt", "r")

for line in rsa.readlines():
    line = line.strip("\n")
    if line.startswith(">"):
        name = line[1:]
    else:
        rsa_matrix = np.zeros((len(line), 1))
        for i in range(len(line)):
            if line[i] == "e":
                rsa_matrix[i, 0] = 1
        os.chdir("/Users/nuno_chicoria/Documents/master_thesis/files/rsa_numpy")
        np.save(name + ".npy", rsa_matrix)
        os.chdir("/Users/nuno_chicoria/Documents/master_thesis/files/rsa_tensor")
        tensor = torch.from_numpy(rsa_matrix)
        torch.save(tensor, name + ".pt")

#For loop for the matrices regarding secundary structure
ss = open("/Users/nuno_chicoria/Documents/master_thesis/datasets/rsa_ss/SSpro.dssp.txt", "r")

for line in ss.readlines():
    line = line.strip("\n")
    if line.startswith(">"):
        name = line[1:]
    else:
        ss_matrix = np.zeros((len(line), 3))
        for i in range(len(line)):
            if line[i] == "C":
                ss_matrix[i, 0] = 1
            elif line[i] == "E":
                ss_matrix[i, 1] = 1
            else:
                ss_matrix[i, 2] = 1
        os.chdir("/Users/nuno_chicoria/Documents/master_thesis/files/ss_numpy")
        np.save(name + ".npy", ss_matrix)
        os.chdir("/Users/nuno_chicoria/Documents/master_thesis/files/ss_tensor")
        tensor = torch.from_numpy(ss_matrix)
        torch.save(tensor, name + ".pt")

