#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for a Feed Forward Neural Network for the prediction of RSA and SS using
frequency table of MSAs as features and RSA and SS as labels.

1. Label Matrix Generation

@author: nuno_chicoria
"""

import os
import numpy as np

#For loop for the matrices regarding relative solvent accessibility
os.chdir("/Users/nuno_chicoria/Documents/master_thesis/rsa_matrices")
rsa = open("/Users/nuno_chicoria/Documents/master_thesis/rsa_ss/ACCpro.dssp.txt", "r")

for line in rsa.readlines():
    line = line.strip("\n")
    if line.startswith(">"):
        name = line[1:]
    else:
        rsa_matrix = np.zeros((len(line), 1))
        for i in range(len(line)):
            if line[i] == "e":
                rsa_matrix[i, 0] = 1
        np.save(name + ".npy", rsa_matrix)

#For loop for the matrices regarding secundary structure
os.chdir("/Users/nuno_chicoria/Documents/master_thesis/ss_matrices")
ss = open("/Users/nuno_chicoria/Documents/master_thesis/rsa_ss/SSpro.dssp.txt", "r")

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
        np.save(name + ".npy", ss_matrix)