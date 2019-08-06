#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:35:43 2018

@author: zhangyaqian
"""
import numpy as np
RANGE = 200

def generateTasks(TaskNum,SkillDim,mode):
    ### 
    
    if(mode=='Gaussian'):
    
        mean = np.zeros(SkillDim) +100
        conv = np.zeros((SkillDim,SkillDim))#
        for i in range(SkillDim):
            conv[i,i]=20
        TaskSkills = np.random.multivariate_normal(mean, conv, (TaskNum))
    else:
        TaskSkills=np.random.randint(1,1+RANGE,size=(TaskNum,SkillDim))



    return TaskSkills.astype('int32')
    
def generateStudents(StudentNum,SkillDim,mode):
    if(mode=='Strong'):
        StudentSkills = (np.random.randint(1+int(RANGE*0.75),1+RANGE,size=(StudentNum,SkillDim)))
    elif(mode=='Weak'):
        StudentSkills = (np.random.randint(1,1+int(RANGE*0.25),size=(StudentNum,SkillDim)))
        
        
    elif(mode=="Gaussian"):
        mean = np.zeros(SkillDim) +100
        conv = np.zeros((SkillDim,SkillDim))#
        for i in range(SkillDim):
            conv[i,i]=20
        StudentSkills = np.random.multivariate_normal(mean, conv, (StudentNum))
    else:
        StudentSkills = (np.random.randint(1,1+RANGE,size=(StudentNum,SkillDim)))
        

    #
    return StudentSkills.astype('int32')



            
        
    