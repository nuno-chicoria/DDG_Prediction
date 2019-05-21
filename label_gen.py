#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Code for a Feed Forward Neural Network for the prediction of RSA and SS using
frequency table of MSAs as features and RSA and SS as labels.

2. Label Matrix Generation

@author: nuno_chicoria
"""

import os
import glob
import numpy as np
from tqdm import tqdm

# For loop for the matrices regarding relative solvent accessibility
rsa = open("/Users/nuno_chicoria/Documents/master_thesis/datasets/rsa_ss/ACCpro.dssp.txt", "r")
os.chdir("/Users/nuno_chicoria/Documents/master_thesis/files/rsa_numpy")

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

# For loop for the matrices regarding secundary structure
ss = open("/Users/nuno_chicoria/Documents/master_thesis/datasets/rsa_ss/SSpro.dssp.txt", "r")
os.chdir("/Users/nuno_chicoria/Documents/master_thesis/files/ss_numpy")

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

# Saving all as a dataset
os.chdir("/Users/nuno_chicoria/Documents/master_thesis/files/rsa_ss_nn/")
window_size = 15
file = open(f"dataset_{window_size}.csv", "w+")

for filepath in tqdm(glob.iglob(f"/Users/nuno_chicoria/Documents/master_thesis/files/msa_{window_size}/*.npy")):
    name = os.path.basename(filepath).partition(".")[0]
    msa = np.load(filepath)
    rsa = np.load(f"/Users/nuno_chicoria/Documents/master_thesis/files/rsa_numpy/{name}.npy")
    ss = np.load(f"/Users/nuno_chicoria/Documents/master_thesis/files/ss_numpy/{name}.npy")
    msa = np.concatenate((msa, rsa), axis = 1)
    msa = np.concatenate((msa, ss), axis = 1)
    with open(f"dataset_{window_size}.csv", "a") as file:
        np.savetxt(file, msa, delimiter = ",")

