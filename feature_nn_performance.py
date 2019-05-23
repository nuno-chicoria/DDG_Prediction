#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for a Feed Forward Neural Network for the prediction of RSA and SS using
frequency table of MSAs as features and RSA and SS as labels.

4. Neural Network Testing and Performance Evaluation

@author: nuno_chicoria
"""

from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from torch.autograd import Variable

"""
TODO
precision
balanced accuracy
specificity
sensitivity
ROC Curve (AUC)
Matthews correlation coefficient
(see which apply to binary and multi class)
"""

#Predictions
rsa_true = []
rsa_pred = []
ss_true = []
ss_pred = []
        
for sample in testset:
    x, yrsa, yss = sample
    yprsa, ypss = model(Variable(x))
    rsa_true.append(yrsa.numpy())
    ss_true.append(yss.numpy())
    rsa_pred.append(yprsa.detach().numpy())
    ss_pred.append(ypss.detach().numpy())
    
#RSA ROC Curve
fpr, tpr, thresholds = metrics.roc_curve(rsa_true, rsa_pred)
optimal_idx = np.argmax(np.abs(tpr - fpr))
rsa_threshold = thresholds[optimal_idx]
roc_auc = metrics.auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw = lw,
         label = f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RSA ROC Curve')
plt.legend(loc="lower right")
plt.show()

#Using the threshold
rsa_binary = []
for i in range(len(rsa_pred)):
    if rsa_pred[i] >= rsa_threshold:
        rsa_binary.append(1)
    else:
        rsa_binary.append(0)

#RSA Metrics
rsa_pres = metrics.precision_score(rsa_true, rsa_binary)
rsa_bal_accu = metrics.balanced_accuracy_score(rsa_true, rsa_binary)
rsa_matthews = metrics.matthews_corrcoef(rsa_true, rsa_binary)

#SS ROC Curve
ss_true_1 = []
for i in range(len(ss_true)):
    ss_true_1.append(ss_true[i][0])
ss_true_2 = []
for i in range(len(ss_true)):
    ss_true_2.append(ss_true[i][1])
ss_true_3 = []
for i in range(len(ss_true)):
    ss_true_3.append(ss_true[i][2])
ss_pred_1 = []
for i in range(len(ss_pred)):
    ss_pred_1.append(ss_pred[i][0])
ss_pred_2 = []
for i in range(len(ss_pred)):
    ss_pred_2.append(ss_pred[i][1])
ss_pred_3 = []
for i in range(len(ss_pred)):
    ss_pred_3.append(ss_pred[i][2])

fpr1, tpr1, thresholds1 = metrics.roc_curve(ss_true_1, ss_pred_1)
optimal_idx1 = np.argmax(np.abs(tpr1 - fpr1))
ss1_threshold = thresholds[optimal_idx1]

fpr2, tpr2, thresholds2 = metrics.roc_curve(ss_true_2, ss_pred_2)
optimal_idx2 = np.argmax(np.abs(tpr2 - fpr2))
ss2_threshold = thresholds[optimal_idx2]

fpr3, tpr3, thresholds3 = metrics.roc_curve(ss_true_3, ss_pred_3)
optimal_idx3 = np.argmax(np.abs(tpr3 - fpr3))
ss3_threshold = thresholds[optimal_idx3]

ss_threshold = (ss1_threshold + ss2_threshold + ss3_threshold) / 3

roc_auc1 = metrics.auc(fpr1, tpr1)
roc_auc2 = metrics.auc(fpr2, tpr2)
roc_auc3 = metrics.auc(fpr3, tpr3)

fpr = [fpr1, fpr2, fpr3]
tpr = [tpr1, tpr2, tpr3]
roc_auc = [roc_auc1, roc_auc2, roc_auc3]

plt.figure()
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color = color, lw = lw,
             label = f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw = lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SS ROC Curve')
plt.legend(loc="lower right")
plt.show()

#Using the threshold and separating into 3 classes
#35519 without any over threshold
ss_binary = []
for i in range(len(ss_pred)):
    temp =[]
    if ss_pred[i][0] >= ss1_threshold:
        temp.append(1)
    else:
        temp.append(0)
    if ss_pred[i][1] >= ss2_threshold:
        temp.append(1)
    else:
        temp.append(0)
    if ss_pred[i][2] >= ss3_threshold:
        temp.append(1)
    else:
        temp.append(0)
    ss_binary.append(temp)

ss1_true = []
ss1_binary = []
for i in range(len(ss_true)):
    ss1_true.append(ss_true[i][0])
    ss1_binary.append(ss_binary[i][0])
ss2_true = []
ss2_binary = []
for i in range(len(ss_true)):
    ss2_true.append(ss_true[i][1])
    ss2_binary.append(ss_binary[i][1])
ss3_true = []
ss3_binary = []
for i in range(len(ss_true)):
    ss3_true.append(ss_true[i][2])
    ss3_binary.append(ss_binary[i][2])

#SS Metrics
#https://scikit-learn.org/stable/modules/multiclass.html
ss1_pres = metrics.precision_score(ss1_true, ss1_binary)
ss2_pres = metrics.precision_score(ss2_true, ss2_binary)
ss3_pres = metrics.precision_score(ss3_true, ss3_binary)

ss1_bal_accu = metrics.balanced_accuracy_score(ss1_true, ss1_binary)
ss2_bal_accu = metrics.balanced_accuracy_score(ss2_true, ss2_binary)
ss3_bal_accu = metrics.balanced_accuracy_score(ss3_true, ss3_binary)

ss1_matthews = metrics.matthews_corrcoef(ss1_true, ss1_binary)
ss2_matthews = metrics.matthews_corrcoef(ss2_true, ss2_binary)
ss3_matthews = metrics.matthews_corrcoef(ss3_true, ss3_binary)

