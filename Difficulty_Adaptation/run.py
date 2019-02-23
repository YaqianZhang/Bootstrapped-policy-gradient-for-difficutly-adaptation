# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 12:18:08 2018

@author: zhangyaqian
"""
import numpy as np

#from PlaySimulation import 

from logger_class import logger

    
def Difficulty_Adaptation(simulator,algo,MaxStep,online_dict,alpha):
 
    TaskSkills = online_dict['TaskSkills']
    students = online_dict['StudentSkills'].copy().astype(float)
    studentNum,dim = students.shape
    taskNum,dim = TaskSkills.shape
    true_skill_id = np.argmax(students,axis = 1)
            

    algo_logger=logger(studentNum,MaxStep)
       
    for i in range(studentNum):
        
        ######### get difficulty ranking
        skillId = np.argmax(online_dict['StudentSkills'][i,:])
        diff_ranking = -online_dict['TaskSkills'][:,skillId]
        
        ######### initial weight
        algo.initialWeights(taskNum,dim,diff_ranking)


        for j in range(MaxStep):
            ################## predict strategy #######################
            
                
            #################### choose a task based on weight ##################
  
            task_ID,action_prob = algo.select_task(skillId,j,diff_ranking)


            #################### compute performance and reward ######################
            grade= simulator.generateStudentPerformance(students[i,true_skill_id[i]],TaskSkills[[task_ID],true_skill_id[i]])
            rwd = simulator.pfmc_to_reward(grade)

                
            #################### update weight ###########################
            algo.update_weights(grade,skillId,diff_ranking,task_ID,alpha,action_prob)
            
            #################### update weight ###########################
            algo_logger.update(i,j,task_ID,grade,rwd)
            
            
    return algo_logger


