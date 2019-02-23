#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:03:20 2019

@author: yaqianzhang
"""
import tensorflow as tf
import numpy as np
class algo():
    def __init__(self,action_dim,para_num,learning_rate):
        with tf.variable_scope(self.name):
                    
            initial_point = 0.1*tf.ones(action_dim)
            self.sy_mean = tf.Variable(initial_point,name="mean")           
            self.sy_ac_na = tf.placeholder(shape=[None,action_dim],name='action',dtype=tf.float32)
            
            
    def process_data(self, paths,std_now,actor_batch_size,critic):
        sample_data={}
        sample_data['action_data']=np.array(paths['action_list'][-actor_batch_size:]).reshape([-1,self.action_dim])
        
        return sample_data
    
class SAC(algo):
    def __init__(self,action_dim,para_num,learning_rate):
        self.name="SAC" # stochastic actor critic
        self.action_dim = action_dim
        algo.__init__(self,action_dim,para_num,learning_rate)
        
        
        with tf.variable_scope(self.name):
            #self.sy_std = tf.placeholder(shape=[action_dim],name='std_ph',dtype=tf.float32)           
            self.sy_std = tf.Variable(0.1*tf.ones(action_dim),name="std_w")
            self.sy_sampled_ac = self.sy_mean+self.sy_std*tf.random_normal(tf.shape(self.sy_mean))
            
            self.sy_features = (self.sy_ac_na-self.sy_mean)
            self.sy_pred_n = tf.placeholder(shape=[None],name="pred_cost",dtype=tf.float32) 
                    
            sy_logprob_n = -tf.reduce_sum(tf.square((self.sy_ac_na-self.sy_mean)/self.sy_std),axis=1)
        
            loss = -tf.reduce_mean(sy_logprob_n*self.sy_pred_n) # Loss function that we'll differentiate to get the policy gradient.
            self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            
    def process_data(self, paths,std_now,actor_batch_size,critic):
           
        sample_data = algo.process_data(self, paths,std_now,actor_batch_size,critic)
        sample_data['std']=std_now

        feature_data = np.array(paths['feature_list'][-actor_batch_size:]).reshape([-1,self.action_dim])
        predict_cost_data = critic.critic_predict(feature_data)           
        sample_data['predict_cost_data']=predict_cost_data
        
        
        return sample_data
    def update_policy(self,sample_data,sess):
        
        [x1]=sess.run([self.update_op],feed_dict={
                                                    self.sy_ac_na:sample_data['action_data'],
                                                    #self.sy_std:sample_data['std'],
                                                    self.sy_pred_n:sample_data['predict_cost_data']
                                                    
                                          })
class PG(algo):
    def __init__(self,action_dim,para_num,learning_rate):
        self.name="SAC" # stochastic actor critic
        self.action_dim = action_dim
        algo.__init__(self,action_dim,para_num,learning_rate)
        
        
        with tf.variable_scope(self.name):
            #self.sy_std = tf.placeholder(shape=[action_dim],name='std_ph',dtype=tf.float32)           
            self.sy_std = tf.Variable(0.1*tf.ones(action_dim),name="std_w") 
            self.sy_sampled_ac = self.sy_mean+self.sy_std*tf.random_normal(tf.shape(self.sy_mean))
            
            self.sy_features = (self.sy_ac_na-self.sy_mean)
            self.sy_adv_n = tf.placeholder(shape=[None],name="adv",dtype=tf.float32) 
                    
            sy_logprob_n = -tf.reduce_sum(tf.square((self.sy_ac_na-self.sy_mean)/self.sy_std),axis=1)
        
            loss = -tf.reduce_mean(sy_logprob_n*self.sy_adv_n) # Loss function that we'll differentiate to get the policy gradient.
            self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            
    def process_data(self, paths,std_now,actor_batch_size,critic):
           
        sample_data = algo.process_data(self, paths,std_now,actor_batch_size,critic)
        sample_data['std']=std_now
        arr = np.array(paths['reward_list'][-actor_batch_size:])
        arr = arr-np.mean(arr)

        sample_data['reward_data'] = arr.reshape([-1]) 
        
        return sample_data
    def update_policy(self,sample_data,sess):
        
        [x1]=sess.run([self.update_op],feed_dict={#self.sy_d_Q:sample_data['dq_data'],
                                                    self.sy_ac_na:sample_data['action_data'],
                                                    #self.sy_std:sample_data['std'],
                                                    self.sy_adv_n:sample_data['reward_data']
                                                    
                                          })



    
class DPG(algo):
    def __init__(self,action_dim,para_num,learning_rate):
        self.name="DPG"
        self.action_dim = action_dim
        algo.__init__(self,action_dim,para_num,learning_rate)
        
        with tf.variable_scope(self.name):
            self.sy_d_Q =tf.placeholder(shape=[None,action_dim],name='d_Q',dtype=tf.float32)
            self.sy_features = (self.sy_ac_na-self.sy_mean)
            self.sy_std = tf.placeholder(shape=[action_dim],name='std_ph',dtype=tf.float32)           
            
            self.sy_sampled_ac = self.sy_mean+self.sy_std*tf.random_normal(tf.shape(self.sy_mean))
            
            
#            loss = -tf.reduce_sum(self.sy_d_Q*self.sy_mean)
#
#            self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#            
            update_amount = learning_rate*tf.reduce_mean(self.sy_d_Q,axis = 0)
            self.update_op = tf.assign(self.sy_mean,self.sy_mean+update_amount)

    def process_data(self, paths,std_now,actor_batch_size,critic):
        sample_data = algo.process_data(self, paths,std_now,actor_batch_size,critic)
        sample_data['dq_data']= np.array(paths['dq_list'][-actor_batch_size:]).reshape([-1,self.action_dim])
        
        
        return sample_data

    def update_policy(self,sample_data,sess):
        
        [x1]=sess.run([self.update_op],feed_dict={self.sy_d_Q:sample_data['dq_data'],
                                                    self.sy_ac_na:sample_data['action_data'],
                                                    
                                          })
        

            
        
class BPG():
    def __init__(self,action_dim,para_num,learning_rate):
        self.name="BPG"
        self.action_dim = action_dim
        algo.__init__(self,action_dim,para_num,learning_rate)
        with tf.variable_scope(self.name):
            self.sy_std = tf.placeholder(shape=[action_dim],name='std_ph',dtype=tf.float32)           
            self.sy_sampled_ac = self.sy_mean+self.sy_std*tf.random_normal(tf.shape(self.sy_mean))
            
            self.sy_d_Q =tf.placeholder(shape=[None,action_dim],name='d_Q',dtype=tf.float32)
            
            
           
            dist=tf.contrib.distributions.Normal(self.sy_mean, self.sy_std)
            smaller_prob_sum = dist.cdf(self.sy_ac_na) # N*A
            larger_prob_sum = 1-smaller_prob_sum #N*A
            sy_logprob_bpg=tf.log(larger_prob_sum*1.0/smaller_prob_sum) # N * A

            self.sy_features = (self.sy_ac_na-self.sy_mean)
            
            #self.sy_features=dist.prob(self.sy_ac_na)/(smaller_prob_sum * larger_prob_sum)*(self.sy_ac_na)
      
#            loss = -tf.reduce_sum(tf.multiply(self.sy_d_Q,sy_logprob_bpg))
#            self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            
            d_logprob=dist.prob(self.sy_ac_na)/(smaller_prob_sum*larger_prob_sum)        
            update_amount = learning_rate*tf.reduce_mean(d_logprob*self.sy_d_Q,axis=0)
            self.update_op = tf.assign(self.sy_mean,self.sy_mean+update_amount)

    def process_data(self, paths,std_now,actor_batch_size,critic):
        sample_data = algo.process_data(self, paths,std_now,actor_batch_size,critic)
        sample_data['dq_data']= np.array(paths['dq_list'][-actor_batch_size:]).reshape([-1,self.action_dim])
        sample_data['std']=std_now
        
        return sample_data


    def update_policy(self,sample_data,sess):
        
        [x1]=sess.run([self.update_op],feed_dict={self.sy_d_Q:sample_data['dq_data'],
                                                    self.sy_ac_na:sample_data['action_data'],
                                                    self.sy_std:sample_data['std']
                                          })
            
    

    
        
    