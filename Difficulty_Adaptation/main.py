#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 19:06:38 2018

@author: zhangyaqian
"""

import time 
from GenerateData import generateTasks, generateStudents 
from algo_class import BPG,PG,Maple,BPG_mpl,RandomPolicy,Bisection
from run import Difficulty_Adaptation
from Plotter import plot_cost,plot_arr_pfmc
from PlaySimulation import Simulator

def generate_online_data(TaskNum,StudentNum,mode=None):
    ## mode control student stype:  "Weak","Strong","Gaussian", "Uniform" default is uniform distribbuion

    SkillDim=1
    online_dict={}
    
    online_dict['TaskSkills'] =generateTasks(TaskNum,SkillDim,mode)
    online_dict['StudentSkills']= generateStudents(StudentNum,SkillDim,mode)
    return online_dict


def get_algo(name,gradeThreshold):
    if(name=="BPG"):
        return BPG(name,gradeThreshold)
    elif(name=="Maple"):
        return Maple(name,gradeThreshold)
    elif(name=="PG"):
        return PG(name,gradeThreshold)
    elif(name=="BPG_mpl"):
        return BPG_mpl(name,gradeThreshold)
    elif(name=="Random"):
        return RandomPolicy(name,gradeThreshold)
    elif(name=="Bisection"):
        return Bisection(name,gradeThreshold)
        
def peform_adaptaion(online_dict,PlaySimulator):
        logger_list=[]
        
        for i in range(len(AlgoNames)):
            start = time.time()
            algo=get_algo(AlgoNames[i],PlaySimulator.target_grade)
            algo_logger=Difficulty_Adaptation(PlaySimulator,algo,SessionNum,online_dict,step_sizes[i])
            logger_list.append(algo_logger)
            print(AlgoNames[i]+"----time elapsed: %.2f" %( time.time()-start))
        return logger_list
    
        




###### simulator of player behavior
#TARGET_GRADE= 0.5
#pfmc_max = 1
PlaySimulator=Simulator()

###### algo
SessionNum = 50
AlgoNames=['Random','Bisection',
           'PG','Maple',  
           'BPG_mpl','BPG',
           ] 
step_sizes=[1,1,
            70,10, 
            400,5000,
            ]#
###### Users
TASK_NUM = 1000
STUDENT_NUM = 500


####### perform adaptation

online_dict = generate_online_data(TASK_NUM,STUDENT_NUM,mode = 'Gaussian')     
logger_list = peform_adaptaion(online_dict,PlaySimulator)           
plot_cost(logger_list,AlgoNames,'Gaussian')

online_dict = generate_online_data(TASK_NUM,STUDENT_NUM,mode = None)     
logger_list = peform_adaptaion(online_dict,PlaySimulator)           
plot_cost(logger_list,AlgoNames,None)

online_dict = generate_online_data(TASK_NUM,STUDENT_NUM,mode = 'Strong')     
logger_list = peform_adaptaion(online_dict,PlaySimulator)  
plot_arr_pfmc(logger_list,AlgoNames,'Strong')

online_dict = generate_online_data(TASK_NUM,STUDENT_NUM,mode = 'Weak')     
logger_list = peform_adaptaion(online_dict,PlaySimulator)  
plot_arr_pfmc(logger_list,AlgoNames,'Weak')






