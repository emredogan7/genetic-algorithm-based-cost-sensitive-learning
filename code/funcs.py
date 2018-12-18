#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 20:07:39 2018

@author: emre
"""

import numpy as np
import pandas as pd
import scipy as sc
from sklearn.neighbors import KNeighborsClassifier
import random
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score

from sklearn.svm import SVC
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import re
from sklearn.metrics import average_precision_score



def classify(X_train,X_test,y_train,y_test):
    
    #model = KNeighborsClassifier(5)
    model = MLPClassifier(alpha=1e-2,hidden_layer_sizes=(100), random_state=1)
    #model = SVC(gamma='auto')
    #model = tree.DecisionTreeClassifier()
    #model = GaussianNB()
    
    model.fit(X_train, y_train)
    
    #y_est_train = model.predict(X_train)
    y_est_test = model.predict(X_test)
    f1List = f1_score(y_test,y_est_test,average=None)
    f1minScore = min(f1List)

    #target_names = ['class 1', 'class 2', 'class 3']
    #report = classification_report(y_test, y_est_test, target_names=target_names)
    accuracy = accuracy_score(y_test, y_est_test)
    return f1minScore,accuracy

def fitness(f1min,accuracy,cost):
    coef1 = 0.6
    coef2 = 0.4
    if (f1min>= 0.7):
        f = coef1*accuracy + coef2*(1-(cost/102.03))
    else:
        f = 0.7*(coef1*accuracy + coef2*(1-(cost/102.03)))
    return f
        
        

def mutation(indices):
    pnt = np.random.randint(len(indices)-1)
    if (indices[pnt] == 0):
        indices[pnt] = 1
    elif (indices[pnt] == 1):
        indices[pnt] = 0

def crossOver(indices1,indices2):
    crossoverPoint = np.random.randint(1,len(indices1)-2)
    offspring1 = [0]*len(indices1)
    offspring2 = [0]*len(indices1)

    offspring1 = indices1[:crossoverPoint] + indices2[crossoverPoint:]
    offspring2 = indices1[crossoverPoint:] + indices2[:crossoverPoint]

    return offspring1, offspring2


def oversampler(data,factor):
    ones_twos = np.zeros((0,22))
    for z in data:
        if (z[21] == 2 or z[21] == 1):
            ones_twos = np.vstack([ones_twos,z])

    newData = data
    for _ in range(factor):
        newData = np.vstack([newData,ones_twos])
        
    random.shuffle(newData)
    return newData

