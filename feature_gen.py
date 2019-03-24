#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Code for a Feed Forward Neural Network for the prediction of RSA and SS using
frequency table of MSAs as features and RSA and SS as labels.

1. Feature Matrix Generation

@author: nuno_chicoria
"""

import glob
import math
import numpy as np
import os
import pickle
import re
import torch
from tqdm import tqdm

#Function for creating the frequency matrix of amino acid pseudo counts for each MSA.
def freqGenerator(filepath):

    msa = open(filepath, "r")
    
    for line in msa.readlines():
        col_size = len(line)
        break
        
    freq_matrix = np.ones((20, col_size))

    aa_pos = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

    msa = open(filepath, "r")
    
    for line in msa.readlines():
        temp = re.split("\n+", line)
        for count, aa in enumerate(temp[0]):
            if aa in aa_pos:
                freq_matrix[aa_pos.index(aa), count] += 1
            else:
                continue

    total_sum = freq_matrix.sum(axis = 0)
    for i in range(0, col_size):
        for j in range(0, 20):
            freq_matrix[j, i] = freq_matrix[j, i]/(total_sum[i] + 20)
    
    return freq_matrix

#Function for creating the feature matrix taking into consideration a sliding window that can be tuned by changing the corresponding variable.
def featureGenerator(matrix, window_size):
    
    feat_matrix = np.zeros((1, 20 * window_size))
    
    frame = math.trunc(window_size / 2)
    
    for i in range(matrix.shape[1]):
        temp_row = np.zeros((1, 1))
        for slide in range(-frame, frame + 1):
            if (i + slide < 0) or (i + slide >= matrix.shape[1]):
                temp_row = np.concatenate((temp_row, np.zeros((1, 20))), axis = 1)
            else:
                temp_row = np.concatenate((temp_row, matrix[:, i + slide].reshape(1, -1)), axis = 1)
        temp_row = np.delete(temp_row, 0, axis = 1)
        feat_matrix = np.concatenate((feat_matrix, temp_row), axis = 0)
        
    feat_matrix = np.delete(feat_matrix, 0, axis = 0)
    
    return feat_matrix

#MAIN METHOD
empty_files = []

for filepath in tqdm(glob.iglob("/Users/nuno_chicoria/Documents/master_thesis/msa_alignments/*.hmmer")):
    name = os.path.basename(filepath).partition("_")[0]
    if os.stat(filepath).st_size == 0:
        name = os.path.basename(filepath).partition("_")[0]
        empty_files.append(name)
    else:
        os.chdir("/Users/nuno_chicoria/Documents/master_thesis/freq_matrices")
        freq_matrix = freqGenerator(filepath)
        np.save(name + ".npy", freq_matrix)
        os.chdir("/Users/nuno_chicoria/Documents/master_thesis/feature_matrices")
        feat_matrix = featureGenerator(freq_matrix, 11)
        np.save(name + ".npy", freq_matrix)
        os.chdir("/Users/nuno_chicoria/Documents/master_thesis/feature_tensors")
        tensor = torch.from_numpy(feat_matrix)
        torch.save(tensor, name + ".pt")

os.chdir("/Users/nuno_chicoria/Documents/master_thesis/rsa_ss")
pickle.dump(empty_files, open( "empty_files.p", "wb"))

