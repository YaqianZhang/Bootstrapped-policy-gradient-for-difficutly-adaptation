#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:16:04 2019

@author: yaqianzhang
"""

import numpy as np
import tensorflow as tf
from critic_OO import CRITIC_MANU

def sample_action(algo,sess,std):

    action = sess.run(algo.sy_sampled_ac,feed_dict={algo.sy_std:std})
    return action
    



def train(algo,env,training_settings):
    [action_dim,para_num,MAX_ITER,actor_batch_size,critic_batch_size]=training_settings

    sess = tf.Session()
    sess.__enter__() 
    tf.global_variables_initializer().run() 


    
    paths={}
    paths['action_list']=[]
    paths['feature_list']=[]
    paths['dq_list']=[]
    paths['reward_list']=[]
    paths['mean_list']=[]

 
    

    critic=CRITIC_MANU(action_dim)
    
    std_initial = 0.1
    
    std_now = std_initial*np.ones(action_dim)
    
    for iter in range(MAX_ITER):
        
    
     
        ## select an action from policy
        
        action = sample_action(algo,sess,std_now)
   
        reward = env.step(action)
        
        [features] = sess.run([algo.sy_features],
                                feed_dict={algo.sy_ac_na:np.array(action).reshape([-1,action_dim]),
                                           algo.sy_std:std_now
                                                     })
        d_Q = critic.theta[1:]
        paths['action_list'].append(action)
        paths['reward_list'].append(reward[0,0])
        
        paths['dq_list'].append(d_Q)
        paths['feature_list'].append(features)
        paths['mean_list'].append(sess.run(algo.sy_mean))

        
        
        if(iter>=critic_batch_size and iter%(actor_batch_size)==0 ):
            
        ## update policy parameters
            sample_data=algo.process_data(paths,std_now,actor_batch_size,critic)
            algo.update_policy(sample_data,sess)
#            
        if(iter > 0 and iter%(critic_batch_size) ==0):
            #print("fit critic (update w)")
            feature_data = np.array(paths['feature_list'][-critic_batch_size:]).reshape([-1,action_dim])
            reward_data = np.array(paths['reward_list'][-critic_batch_size:]).reshape([-1])

            
            critic.critic_training(feature_data,reward_data)


    sess.close()
    return paths,critic
    
