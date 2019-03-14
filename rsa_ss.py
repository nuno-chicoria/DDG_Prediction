#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Code for a Feed Forward Neural Network for the prediction of RSA and SS using
frequency table of MSAs as features and RSA and SS as labels.

@author: nuno_chicoria
"""

import numpy as np
import re
import os
import glob

#Function for creating the frequency matrix of amino acid pseudo counts for
#each MSA.
def freqGenerator(filepath):

    msa = open(filepath, "r")
    
    for line in msa.readlines():
        col_size = len(line)
        break
        
    freq_matrix = np.ones((20, col_size))

    aa_pos = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M",
              "F", "P", "S", "T", "W", "Y", "V"]

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
            
    name = os.path.basename(filepath).partition("_")[0] + "_freqmatrix.npy"
    np.save(name, freq_matrix)
    
    return freq_matrix

#Function for creating the feature matrix taking into consideration a sliding
#window that can be tuned by changing the corresponding variable.
def featureGenerator(matrix, windowSize):
    
    
    
    return "Hello"

#MAIN METHOD
empty_files = []

for filepath in glob.iglob("/Users/nuno_chicoria/Documents/master_thesis/msa_alignments/*.hmmer"):
    if os.stat(filepath).st_size == 0:
        name = os.path.basename(filepath).partition("_")[0]
        empty_files.append(name)
    else:
        os.chdir("/Users/nuno_chicoria/Documents/master_thesis/freq_matrices")
        freqGenerator(filepath)
        os.chdir("/Users/nuno_chicoria/Documents/master_thesis/feature_matrices")
        featureGenerator(filepath)