#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 17:30:03 2019

@author: yaqianzhang
"""
import numpy as np

def sign(x):
    sign_mat  = np.zeros(len(x))
    sign_mat[x<0]=-1
    sign_mat[x>0]=1
    return sign_mat

def softmax(vec,j):
    #nn=vec-np.mean(vec)

    
    para = 0#1.0/ np.sqrt(j+1)

    noise = np.random.rand(1)*2-1  ###[-1, 1]
    nn = (1-para) * vec+ (para)*noise

    
    nn = nn - np.max(nn)
    #print np.max(nn)

    
    nn1 = np.exp(nn)
    #print np.max(nn1)
    #nn1 = 1/(1+np.exp(-nn))
    vec_prob = nn1*1.0/np.sum(nn1)
    return vec_prob

def softmax0(vec):
    #

    #nn = np.exp(vec)
    vec_sum = np.sum(vec)#np.sum(nn)
    prob = vec*1.0/(vec_sum)
    
    return prob