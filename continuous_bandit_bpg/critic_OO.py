#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:02:52 2019

@author: yaqianzhang
"""

import numpy as np
from sklearn import  linear_model


class CRITIC_MANU():
    ## linear regression 
    def __init__(self,input_num):
        
        self.theta=0.01*np.ones(input_num+1)
        
        
    def critic_training(self,x,y):
       
        n = x.shape[0]
        bias = np.ones([n,1])
        x_train_b = np.append(bias,x,axis = 1)
        
        xx= np.matmul(x_train_b.T,x_train_b)
        
        theta1 =  np.linalg.pinv(xx)
        theta2 = np.matmul(theta1,x_train_b.T)
        theta = np.matmul(theta2,y) 
        
        #print("Error0",np.mean(np.square(np.matmul(x_train_b,theta)-y)))
        self.theta = theta
    def critic_predict(self,x):
        n = x.shape[0]
        bias = np.ones([n,1])
        x_train_b=np.append(bias,x,axis = 1)
        return np.matmul(x_train_b,self.theta)
    
class CRITIC_SCI(): 
    def __init__(self,input_num):
        
        self.theta=0.01*np.ones(input_num+1) 
        self.regr = linear_model.LinearRegression() 
    
    def critic_training(self,x,y):
        
        
        self.regr.fit(x, y)

        self.theta=np.append([self.regr.intercept_],self.regr.coef_)
    def critic_predict(self,x):
        y_pred= self.regr.predict(x)
        return y_pred
      
