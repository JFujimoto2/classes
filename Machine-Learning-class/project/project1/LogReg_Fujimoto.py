#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:09:06 2019
@author: Jumpei Fujimoto
"""

import numpy as np
from scipy.optimize import minimize

class LogReg:
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.N = len(y)
        self.classes = np.unique(self.y)
        
        def NLL(beta):
            ones = np.ones((self.N, 1))
            X_ = np.hstack((ones, self.X))
            z = np.dot(X_, beta)
            z = z.reshape(-1,)
            p = (1/(1 + np.exp(z)))
            pi = np.hstack((1- p[(np.where(self.y == self.classes[1]))], p[(np.where(self.y == self.classes[0]))]))
            return sum(-np.log(pi))
            
        
        beta_guess = np.zeros(self.X.shape[1] + 1)
        min_results = minimize(NLL, beta_guess)
        self.coefficients = min_results.x
        self.loss = round(NLL(self.coefficients),4)
        self.accuracy = round(self.score(X, y),4)
        ##self.accuracy = round(self.score(self.X, self.y),4)

        
    def predict_proba(self, X):
        X = np.array(X)
        ones = np.ones((X.shape[0], 1))
        X_ = np.hstack((ones, X))
        z = np.dot(X_, self.coefficients)
        z = z.reshape(-1,)
        return (1/(1 + np.exp(-z)))
        
    def predict(self, X, t = 0.5):
        self.X = np.array(X)
        predict = np.where(self.predict_proba(X) <t, self.classes[0], self.classes[1])
        return predict

        
    def score(self, X, y, t = 0.5):
        self.X = np.array(X)
        self.y = np.array(y)
        
        result = np.sum(self.predict(X, t) == self.y) / len(self.y)
        return result
        
    def summary(self):
        print(" +----------------------------------+")
        print(" | Logistic Regression Summary  |")
        print(" +----------------------------------+")
        print("Number of training observations: " + str(self.N))
        print("Coefficient estimate: " + str(self.coefficients))
        print("Negative Log-Likelihood: " + str(self.loss))
        print("Accuracy: " + str(self.accuracy))
        
    def confusion_matrix(self, X, y, t = 0.5):
        self.matrix = np.zeros(shape = (2,2), dtype = 'int')
        res = np.where(self.predict_proba(X) <t, self.classes[0], self.classes[1])
        
        for i in range(0, len(res)):
            if res[i] == self.classes[0]:
                if self.y[i] == self.classes[0]:
                    self.matrix[0, 0] += 1
                else:
                    self.matrix[1, 0] += 1
            else:
                if self.y[i] == self.classes[1]:
                    self.matrix[1, 1] += 1
                else:
                    self.matrix[0, 1] += 1
        
        return self.matrix
        
    def precision_recall(self, X, y, t=0.5):
        self.X = np.array(X)
        self.y = np.array(y)
        self.confusion_matrix(X, y, t)
        re0 = round(self.matrix[0, 0] / (self.matrix[0, 0] + self.matrix[0, 1]),4)
        pr0 = round(self.matrix[0, 0] / (self.matrix[0, 0] + self.matrix[1, 0]),4)
        re1 = round(self.matrix[1, 1] / (self.matrix[1, 1] + self.matrix[1, 0]),4)
        pr1 = round(self.matrix[1, 1] / (self.matrix[1, 1] + self.matrix[0, 1]),4)
        
        print("Class: " + str(self.classes[0]))
        print("   Precision = " + str(pr0))
        print("   Recall    = " + str(re0))
        print("Class: " + str(self.classes[1]))
        print("   Precision = " + str(pr1))
        print("   Recall    = " + str(re1))
   
        
   
        
        
        
    

        
        
        


