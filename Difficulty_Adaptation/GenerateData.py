#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:35:43 2018

@author: zhangyaqian
"""
import numpy as np
RANGE = 200

def generateTasks(TaskNum,SkillDim):
    ### 
    TaskSkills=np.random.randint(1,1+RANGE,size=(TaskNum,SkillDim))

    return TaskSkills.astype('int32')
    
def generateStudents(StudentNum,SkillDim,mode="Normal"):
    if(mode=='Weak'):
        StudentSkills = (np.random.randint(1+int(RANGE*0.75),1+RANGE,size=(StudentNum,SkillDim)))
    elif(mode=='Strong'):
        StudentSkills = (np.random.randint(1,1+int(RANGE*0.25),size=(StudentNum,SkillDim)))
    else:
        StudentSkills = (np.random.randint(1,1+RANGE,size=(StudentNum,SkillDim)))
        

#
    return StudentSkills.astype('int32')



            
        
    