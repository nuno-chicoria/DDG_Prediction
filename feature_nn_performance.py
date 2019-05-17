#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for a Feed Forward Neural Network for the prediction of RSA and SS using
frequency table of MSAs as features and RSA and SS as labels.

4. Neural Network Testing and Performance Evaluation

@author: nuno_chicoria
"""

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle

#import trainset, testset, model

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
fpr, tpr, thresholds = roc_curve(rsa_true, rsa_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RSA ROC Curve')
plt.legend(loc="lower right")
plt.show()

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

fpr1, tpr1, thresholds1 = roc_curve(ss_true_1, ss_pred_1)
fpr2, tpr2, thresholds2 = roc_curve(ss_true_2, ss_pred_2)
fpr3, tpr3, thresholds3 = roc_curve(ss_true_3, ss_pred_3)

roc_auc1 = auc(fpr1, tpr1)
roc_auc2 = auc(fpr2, tpr2)
roc_auc3 = auc(fpr3, tpr3)

fpr = [fpr1, fpr2, fpr3]
tpr = [tpr1, tpr2, tpr3]
roc_auc = [roc_auc1, roc_auc2, roc_auc3]

plt.figure()
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SS ROC Curve')
plt.legend(loc="lower right")
plt.show()

#rsa_matthews = matthews_corrcoef(rsa_true, rsa_pred)
#rsa_conf = confusion_matrix(rsa_true, rsa_pred)
#rsa_report = classification_report(y_true = rsa_true, y_pred = rsa_pred)

#ss_matthews = matthews_corrcoef(ss_true, ss_pred)
#ss_conf = confusion_matrix(ss_true, ss_pred)
#ss_report = classification_report(y_true = ss_true, y_pred = ss_pred)

