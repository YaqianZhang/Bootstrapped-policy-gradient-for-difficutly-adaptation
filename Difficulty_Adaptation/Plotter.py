#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 14:52:21 2019

@author: yaqianzhang
"""

import matplotlib.pyplot as plt
import numpy as np


def PlotResult(logger_list,names):
    
    algo_num = len(logger_list)
    legend_name = names
    legends=['-o','.-','-*','-<','-^','-v','-.*','-o','.-','*-','<-','-o','.-']
    #legends2=['-.o','.-.','-.*','-<','-^','-v','-.*','-o','.-','*-','<-','-o','.-']
  
    

    plt.figure(figsize=(5,4))#figsize=(8,6)
    ax = plt.subplot(111)
    for i in range(algo_num):
        #rewards = pfmc_to_reward(logger_list[i].arr_pfmc,threshold,pfmc_max)
        rewards = logger_list[i].arr_rwd
        time_reward = np.mean(rewards,axis=0)
        plt.plot(time_reward,legends[i])
        

    
    ax.legend(legend_name[:algo_num],loc='center right', bbox_to_anchor=(1, 0.55),
          ncol=2, fancybox=True, shadow=True)
    #plt.legend(legend_name[:num])
    plt.xlabel("Time step")
    plt.ylabel("Cost")
    #plt.yticks(np.arange(3), ('10^1', '10^0', '10^-1'))
    plt.title("Data with Uniform Distribution")
    #plt.ylim(0,1)
    plt.show()

#          
        #####################  action quality
   


#    
    plt.figure(figsize=(5,4))
    ax=plt.subplot(111)
    
    for i in range(algo_num):
          sr = logger_list[i].arr_pfmc
          time_match = np.mean(sr,axis = 0)
          plt.plot(time_match,legends[i])
          #temp_name.append(names[i]+' For WS')
    #plt.legend(legend_name[:num],loc=1)
    plt.legend(legend_name[:algo_num],loc='upper right', bbox_to_anchor=(1, 1.0),
          ncol=2, fancybox=True, shadow=True)
    #plt.ylabel("Felt Difficulty")
    plt.ylim(0,1)
    plt.yticks([0, 0.5,1.0], ["",'',''])#["Hard", "Suitable","Easy"])
    plt.title('Weak Students')
    
    plt.xlabel("Time step")
  
    

    
