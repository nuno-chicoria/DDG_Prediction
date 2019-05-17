#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for cleaning and extracting all single PDB codes from the S2648 dataset
for use in the MSA.

@author: nuno_chicoria
"""

import os
import re

os.chdir("/Users/nuno_chicoria/Documents/master_thesis/datasets_others")

s2648 = open("dataset_S2648.csv", "r")

test_dict = {}
for line in s2648.readlines():
        temp = re.split(",", line)
        if temp[0][4] in test_dict.keys():
            test_dict[temp[0][4]].append(temp[0][:4])
        else:
            test_dict[temp[0][4]] = []
            test_dict[temp[0][4]].append(temp[0][:4])

C = set(test_dict["C"])
A = set(test_dict["A"])
U = set(test_dict["U"])
I = set(test_dict["I"])
X = set(test_dict["X"])
B = set(test_dict["B"])
_1 = set(test_dict["1"])

f = open("C.txt", "w+")
for entry in C:
    f.write(entry + ",")
f = open("A.txt", "w+")
for entry in A:
    f.write(entry + ",")
f = open("U.txt", "w+")
for entry in U:
    f.write(entry + ",")
f = open("I.txt", "w+")
for entry in I:
    f.write(entry + ",")
f = open("X.txt", "w+")
for entry in X:
    f.write(entry + ",")
f = open("B.txt", "w+")
for entry in B:
    f.write(entry + ",")
f = open("_1.txt", "w+")
for entry in _1:
    f.write(entry + ",")
