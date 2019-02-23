#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:35:24 2019

@author: yaqianzhang
"""

import numpy as np

class SYSTEM():
    def __init__(self,action_dim,fixed_matrix=False):
        self.beta=0.99 ## randomness
        TARGET = -4
        self.target_action = np.ones(action_dim)*TARGET
        if(fixed_matrix):
            self.matrix = np.diag(np.ones(action_dim)*0.1)
        else:
        #target_action = np.array([-100,-400,-400,200])
            A= np.random.rand(action_dim,action_dim)
            B=(A+A.T)/2
            _, s, V = np.linalg.svd(B)
            c_matrix = np.zeros((action_dim,action_dim))
            np.fill_diagonal(c_matrix,np.ones(action_dim)*0.1)
            B_new = V.T.dot(c_matrix).dot(V)
            self.matrix=B_new
        
        
    def step(self,action):       
        u = np.random.rand()
        dis_vector = (action-self.target_action).reshape([-1,1])
        cost = np.matmul(dis_vector.T,self.matrix).dot(dis_vector)
        cost=cost*self.beta + u*(1-self.beta)
        reward = -cost 
        return reward
        
