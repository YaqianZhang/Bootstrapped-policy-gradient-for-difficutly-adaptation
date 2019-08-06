#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 17:05:37 2019

@author: yaqianzhang
"""
import numpy as np

from utilities import softmax
#class D_ALGO():
class Bisection():
    def __init__(self,name,gradeThreshold):
        self.name = name
        self.gradeThreshold=gradeThreshold
        self.left = 0
        self.right=0
    def initialWeights(self,taskNum,num_cluster,rltv_pfmc):
        self.left = np.min(-rltv_pfmc)
        self.left0=self.left
        self.right = np.max(-rltv_pfmc)
        self.right0=self.right
        #r = int(taskNum/MaxStep)
    def select_task(self,skillId,steps,rltv_pfmc):
#   
#                #if(j==0 or grade > gradeThreshold):## too easy
#                q_id=j*r
#             
#                task_ID = np.argsort(rltv_pfmc)[::-1][q_id]
        if(self.left < self.right and self.left>=self.left0 and self.right<=self.right0):
            diff_idx = int((self.left+self.right)/2)
        else:
            diff_idx = int((self.left+self.right)/2)
            
            #print left,right,diff_idx
            
        #task_ID = task_sort[task_idx]
        if(len(np.where(rltv_pfmc==-diff_idx)[0])>0):
            task_ID = np.where(rltv_pfmc==-diff_idx)[0][0]
        else:
            task_ID = 0
        
        return task_ID,diff_idx
            #print task_ID
        
        #print task_ID
    def update_weights(self,grade,skillId,diff_ranking,task_ID,para,diff_idx):
        if(grade>self.gradeThreshold):#too easy
            self.left = diff_idx#task_ID
        elif(grade<self.gradeThreshold):## too hard
            self.right = diff_idx#task_ID



class S_ALGO():
    ## base class for difficulty adaptation schochastic algo
    def __init__(self,name,gradeThreshold):
        self.name=name
        self.taskNum=0#taskNum
        self.weight=0#100*np.ones([taskNum,num_cluster])
        self.gradeThreshold = gradeThreshold
        
    def initialWeights(self,taskNum,num_cluster,diff_ranking):
            #weights = np.transpose(np.ones([num_cluster,taskNum])*advice_vec)
            self.weights = 100*np.ones([taskNum,num_cluster])
            self.taskNum = taskNum
    def select_task(self,skillId,steps,diff_ranking):
        weights_repeat  = self.weights[:,skillId] 
        
        #weights_repeat = w_single*repeat_flag
        
        action_prob = softmax(weights_repeat,steps) # 200*1
        action_noise = action_prob


        task_ID = np.random.choice(range(self.taskNum), 1, replace=False, p=action_noise)
        return task_ID, action_prob
    
    def _computeRewardUpdate(self,grade,para):
        rwd=(-np.abs(grade-self.gradeThreshold)*1.0)
    
 
        return rwd*para
    def _updateActionSetWeight(self,skillId,set_idx,rwd,action_prob):
    
        if(np.sum(action_prob[set_idx])>0):
            ww2 = 1.0*action_prob[set_idx]/np.sum(action_prob[set_idx]) 
            self.weights[set_idx,skillId]+= rwd*ww2
            self.weights[:,skillId] -= rwd * action_prob 

class RandomPolicy(S_ALGO):
    def update_weights(self,grade,skillId,diff_ranking,task_ID,para,action_prob):
        pass
        
class BPG(S_ALGO):
    def _constructBetterWorseActionSet(self,diff_ranking,task_ID,grade):
        idx_same = diff_ranking==diff_ranking[task_ID]
        idx_good = []
        idx_bad = []
        ## promising set
        if(grade<self.gradeThreshold):## TOO hard
            idx_good = diff_ranking>diff_ranking[task_ID]
            idx_bad = diff_ranking<diff_ranking[task_ID]
        
        elif(grade>self.gradeThreshold):
            idx_good = diff_ranking<diff_ranking[task_ID]
            idx_bad = diff_ranking>diff_ranking[task_ID]
        else:
            idx_good = []#diff_ranking==diff_ranking[task_ID]
            idx_bad = diff_ranking != diff_ranking[task_ID]
        return idx_same,idx_good,idx_bad

    def update_weights(self,grade,skillId,diff_ranking,task_ID,para,action_prob):
        update_times_rwd = self._computeRewardUpdate(grade,para)
        [idx_same,idx_good,idx_bad]=self._constructBetterWorseActionSet(diff_ranking,task_ID,grade)

        idx = idx_good 
        rwd = np.abs(update_times_rwd)
        self._updateActionSetWeight(skillId,idx,rwd,action_prob)

        
        
class Maple(S_ALGO):
    def update_weights(self,grade,skillId,diff_ranking,task_ID,para,action_prob):  
        update_times_rwd = self._computeRewardUpdate(grade,para)
        
        idx = diff_ranking<diff_ranking[task_ID] ## hard one
       
        if(grade<=self.gradeThreshold):## hard, decrease harder
            
            self.weights[idx,skillId]-= np.abs(update_times_rwd)
            self.weights[:,skillId] += action_prob * 0.1#update_times_rwd
        else: ##
            self.weights[idx,skillId]+= np.abs(update_times_rwd)
            self.weights[:,skillId] -= action_prob * 0.1#update_times_rwd

         
class PG(S_ALGO):
    def update_weights(self,grade,skillId,diff_ranking,task_ID,para,action_prob):  
        update_times_rwd = self._computeRewardUpdate(grade,para)
        idx_same = diff_ranking==diff_ranking[task_ID]
        
        #[idx_same,idx_good,idx_bad]=self._computeRelatedQuestionIndex(diff_ranking,task_ID,grade)
        #self.weights=updateActionSetWeight(self.weights,skillId,idx_same,update_times_rwd,action_prob)
        self._updateActionSetWeight(skillId,idx_same,update_times_rwd,action_prob)
class BPG_mpl(S_ALGO):
    def update_weights(self,grade,skillId,diff_ranking,task_ID,para,action_prob):
        update_times_rwd = self._computeRewardUpdate(grade,para)
        idx = diff_ranking<diff_ranking[task_ID] ## hard one
        
        
        if( grade<=self.gradeThreshold ):
             ## decrease
            rwd = -np.abs(update_times_rwd)
            self._updateActionSetWeight(skillId,idx,rwd,action_prob)

            ## increase good questions
        if(grade >self.gradeThreshold ):

            rwd = np.abs(update_times_rwd)
            self._updateActionSetWeight(skillId,idx,rwd,action_prob)

    
