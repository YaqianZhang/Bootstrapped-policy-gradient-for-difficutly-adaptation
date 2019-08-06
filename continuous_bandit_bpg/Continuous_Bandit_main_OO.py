#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:20:38 2019

@author: yaqianzhang
"""

import time



from ControlSystem_OO import SYSTEM
from RL_ALGO_OO import DPG,BPG,SAC,PG
from training_process_OO import train
from PlotResult import showResult


    
def get_algo(algo_name,action_dim,para_num,learning_rate):
    if(algo_name == "DPG"):
        algo=DPG(action_dim,para_num,learning_rate)
    elif(algo_name == "BPG"):
        algo=BPG(action_dim,para_num,learning_rate)
    elif(algo_name == "SAC"):
        algo=SAC(action_dim,para_num,learning_rate)
    elif(algo_name == "PG"):
        algo=PG(action_dim,para_num,learning_rate)
    return algo



def main(action_dim =10):
    
    
    
### define problem
    para_num = action_dim 
    actor_batch_size = action_dim*2
    critic_batch_size =action_dim*2
    MAX_ITER=10000
    
    
    training_settings=[action_dim,para_num,MAX_ITER,actor_batch_size,critic_batch_size]
    
    
    env=SYSTEM(action_dim)
    
    
    names=[]
    critic_list=[]
    
    paths_algos=[]
    
    
    algo_names = ['PG','DPG','BPG']
    
    learning_rates = [0.03,0.1,0.008] #-4 20
    for i in range(len(algo_names)):
        start_time=time.time()
        
        
        algo=get_algo(algo_names[i],action_dim,para_num,learning_rates[i])
        
        
        paths,critic = train(algo,env,training_settings)
        paths_algos.append(paths)
    
    
        names.append(algo_names[i])
        critic_list.append(critic)
    
        print ((algo_names[i]+": reward is %.2e, time is %.2f") % (paths['reward_list'][-1], time.time()-start_time))
    
    showResult(actor_batch_size,action_dim,paths_algos,names)


if __name__ == "__main__":
    main(action_dim=10)
    main(action_dim=60)    