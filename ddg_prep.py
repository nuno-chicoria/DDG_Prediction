#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for the feature matrix of DDG prediction

@author: nuno_chicoria
"""

import pandas as pd

seq = open("/Users/nuno_chicoria/Documents/master_thesis/datasets_others/S2648/s2648_fasta.txt", "r")

seq_dict = {}

for line in seq:
    if line.startswith(">"):
        temp = line.rstrip("\n")
        name = temp[1:5] + temp[6]
        seq_dict[name] = ""
    else:
        seq_dict[name] += line.rstrip("\n")

rsa = open("/Users/nuno_chicoria/Documents/master_thesis/files/s2648_rsa_ss/s2648_rsa.txt", "r")

rsa_dict = {}

for line in rsa:
    if line.startswith(">"):
        temp = line.rstrip("\n")
        name = temp[1:6]
        rsa_dict[name] = ""
    else:
        rsa_dict[name] += line.rstrip("\n")

ss = open("/Users/nuno_chicoria/Documents/master_thesis/files/s2648_rsa_ss/s2648_ss.txt", "r")

ss_dict = {}

for line in ss:
    if line.startswith(">"):
        temp = line.rstrip("\n")
        name = temp[1:6]
        ss_dict[name] = ""
    else:
        ss_dict[name] += line.rstrip("\n")
        
dataset = pd.read_csv("/Users/nuno_chicoria/Documents/master_thesis/datasets_others/S2648/dataset_S2648.csv")

X = []
y = []
skip = ["2A01A", "1AMQA", "1ZG4A", "2OCJA"] #1st is deprecrated, others mutation is out of sequence

for a in range(len(dataset)):
    name = dataset["PDB_CHAIN"][a]
    temp_features = []
    if name in skip:
        continue
    else:
        if int(dataset["POSITION"][a]) == 1:
            temp_rsa = rsa_dict[name][int(dataset["POSITION"][a]) - 1 : int(dataset["POSITION"][a]) + 1]
            temp_ss = ss_dict[name][int(dataset["POSITION"][a]) - 1 : int(dataset["POSITION"][a]) + 1]
            temp_features.append(0)
            for i in temp_rsa:
                if i == "e":
                    temp_features.append(1)
                else:
                    temp_features.append(0)
            temp_features.append(0)
            for i in temp_ss:
                if i == "H":
                    temp_features.append(1)
                elif i == "C":
                    temp_features.append(2)
                else:
                    temp_features.append(3)
        elif int(dataset["POSITION"][a]) == len(seq_dict[name]):
            temp_rsa = rsa_dict[name][int(dataset["POSITION"][a]) - 2 : int(dataset["POSITION"][a])]
            temp_ss = ss_dict[name][int(dataset["POSITION"][a]) - 2 : int(dataset["POSITION"][a])]
            for i in temp_rsa:
                if i == "e":
                    temp_features.append(1)
                else:
                    temp_features.append(0)
            temp_features.append(0)
            for i in temp_ss:
                if i == "H":
                    temp_features.append(1)
                elif i == "C":
                    temp_features.append(2)
                else:
                    temp_features.append(3)
            temp_features.append(0)
        else:
            temp_rsa = rsa_dict[name][int(dataset["POSITION"][a]) - 2 : int(dataset["POSITION"][a]) + 1]
            temp_ss = ss_dict[name][int(dataset["POSITION"][a]) - 2 : int(dataset["POSITION"][a]) + 1]
            if len(temp_rsa) != 3 or len(temp_ss) != 3:
                print(name)
                print(dataset["POSITION"][a])
            for i in temp_rsa:
                if i == "e":
                    temp_features.append(1)
                else:
                    temp_features.append(0)
            for i in temp_ss:
                if i == "H":
                    temp_features.append(1)
                elif i == "C":
                    temp_features.append(2)
                else:
                    temp_features.append(3)
        X.append(temp_features)
        y.append(dataset["EXP_DDG"][a])

