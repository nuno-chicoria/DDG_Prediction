#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for a Feed Forward Neural Network for the prediction of RSA and SS using
frequency table of MSAs as features and RSA and SS as labels.

@author: nuno_chicoria
"""

import numpy as np
import re

#Function for creating the frequency matrix of amino acid pseudo counts for
#each MSA.
def FreqMatrix(filepath):

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
    
    return freq_matrix

#Function for creating the feature matrix taking into consideration a sliding
#window that can be tuned by changing the corresponding variable.
def FeatureMatrix(FreqMatrix, SlidingWindow):
    
    return "Hello"

