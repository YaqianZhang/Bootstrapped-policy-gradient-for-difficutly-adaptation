# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 12:19:27 2018

@author: zhangyaqian
"""

import numpy as np

class Simulator():
    def __init__(self,target_grade = 0.5, max_grade =1):
        self.target_grade = 0.5
        self.max_grade = 1

    def pfmc_to_reward(self,pfmc):

        res=np.abs(pfmc-self.target_grade)
    
        return res

    def _computeProb(self,t_skill,s_skill):
        #return (20-t_skill)*1.0/20
        u = np.random.rand()
        theta = 1
        beta =0.99
        prob=1.0/(1.0+np.exp(theta*(t_skill-s_skill)) )
        prob_success = beta * prob + (1-beta)*u
        
        return prob_success
    
    def generateStudentPerformance(self, student, task):
        ## student: dim*1
        ##Task: numTask *dim
        #print task.shape, student.shape
        numTask = task.shape[0]
        prob_success = np.zeros(numTask)
    
        for i in range(numTask):      
            #print task[skillId],student[skillId]
            prob_success[i]=self._computeProb(task[i],student)
        
        return prob_success

