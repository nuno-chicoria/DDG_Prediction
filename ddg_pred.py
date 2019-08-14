#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for the DDG prediction using SVC, SVR and PCA and respective performance testing

7. DDG Prediction Algorithms and Performance Testing

@author: nuno_chicoria
"""

import math
import numpy as np
import pandas as pd
import scipy
import statistics as st
from sklearn import svm
from sklearn.decomposition import PCA
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt

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
y_cat = []
aa_pos = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
skip = ["2A01A", "1AMQA", "1ZG4A", "2OCJA"] #1st is deprecrated, other mutations are out of sequence
window_size = 5
frame= math.trunc(window_size / 2)

for a in range(len(dataset)):
    name = dataset["PDB_CHAIN"][a]
    temp_features = []
    if name in skip:
        continue
    else:
        freq_table = np.load(f"/Users/nuno_chicoria/Documents/master_thesis/files/s2648_freq/{name}.npy")
        position = int(dataset["POSITION"][a]) - 1
        for slide in range(-frame, frame + 1):
            if (position + slide < 0) or (position + slide >= len(seq_dict[name])):
                temp_features.append(0)
            else:
                temp_features.append(rsa_dict[name][position + slide])
        for slide in range(-frame, frame + 1):
            if (position + slide < 0) or (position + slide >= len(seq_dict[name])):
                temp_features.append(0)
            else:
                temp_features.append(ss_dict[name][position + slide])
        for slide in range(-frame, frame + 1):
            if (position + slide < 0) or (position + slide >= len(seq_dict[name])):
                listofzeros = [0] * 20
                temp_features = temp_features + listofzeros
            else:
                temp_features = temp_features + freq_table[:, position + slide].tolist()
        freq = freq_table[aa_pos.index(dataset["MUTANT_RES"][a]), int(dataset["POSITION"][a]) - 1]
        odds = freq / (1 - freq)
        temp_features.append(math.log(odds))
        X.append(temp_features)
        if dataset["EXP_DDG"][a] < -1.0:
            y_cat.append(0)
        else:
            y_cat.append(1)
        y.append(dataset["EXP_DDG"][a])
                
for i in range(len(X)):
    for j in range(len(X[i])):
        if X[i][j]  == "e":
            X[i][j] = 1
        if X[i][j]  == "-":
            X[i][j] = 0
        if X[i][j]  == "H":
            X[i][j] = 1
        if X[i][j]  == "C":
            X[i][j] = 2
        if X[i][j]  == "E":
            X[i][j] = 3

clf = svm.SVC(C = 1, gamma = "auto")
clf.fit(X, y_cat)
#categorical
clf = svm.SVC(C = 1, gamma = "auto")
scores = cross_val_score(clf, X, y_cat, cv = 10, scoring = "roc_auc")
print(f"score: {st.mean(scores)}")

#regression
clf = svm.SVR(gamma = 'auto', C = 1)
r2 = cross_val_score(clf, X, y, scoring = "r2", cv = 10)
mse = cross_val_score(clf, X, y, scoring = "neg_mean_squared_error", cv = 10)
exp_variance = cross_val_score(clf, X, y, scoring = "explained_variance", cv = 10)
print(f"r2: {st.mean(r2)}, mse: {st.mean(mse)}, var: {st.mean(exp_variance)}")


#SVR
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = svm.SVR(gamma = 'auto', C = 1)
clf.fit(X_train, y_train)
X_test = np.array(X_test)
y_pred = clf.predict(X_test)
corr, p_value = scipy.stats.pearsonr(y_pred, y_test)
print(f"score: {corr}")

# Plot
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('test')
plt.ylabel('pred')
plt.show()

#PCA
pca = PCA(n_components = 1)
principalComponents = np.concatenate(pca.fit_transform(X), axis = 0)
corr, pvalue = scipy.stats.pearsonr(principalComponents, y)
rmse = metrics.mean_squared_error(y, principalComponents)
var = metrics.explained_variance_score(y, principalComponents)
print(f"Pearson: {corr}, MSE: {rmse}, var: {var}")

