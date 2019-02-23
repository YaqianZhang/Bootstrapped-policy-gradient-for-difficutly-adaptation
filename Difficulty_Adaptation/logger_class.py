#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 11:56:40 2019

@author: yaqianzhang
"""

import numpy as np
class logger():
    def __init__(self,studentNum,MaxStep):
        self.arr_pfmc=np.zeros((studentNum,MaxStep))
        self.arr_rwd = np.zeros((studentNum,MaxStep))
        self.arr_action = np.zeros((studentNum,MaxStep)).astype(int)
        #self.final_ids=np.zeros((studentNum,int(MaxStep/est_step-1)))
    
        #steps=[1,3,6,9,12,15,30]
        #steps=range(MaxStep)
    
    #step_nn=0
        #self.final_weights = np.zeros((studentNum,taskNum,len(steps)))
    def update(self,i,j,task_ID,grade,rwd):
        self.arr_action[i,j]=int(task_ID)   
        self.arr_pfmc[i,j] = grade
        self.arr_rwd[i,j] = rwd