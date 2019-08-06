#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 14:52:21 2019

@author: yaqianzhang
"""

import matplotlib.pyplot as plt
import numpy as np

legends=['-o','.-','-*','-<','-^','-v','-.*','-o','.-','*-','<-','-o','.-']
def plot_cost(logger_list,names,mode=None):
#    params = {'legend.fontsize': 20,
#          'axes.titlesize': 24,
#          'axes.labelsize': 20,
#          'lines.linewidth' : 3,
#          'lines.markersize' : 10,
#          'xtick.labelsize': 16,
#          'ytick.labelsize': 16}
#    plt.rcParams.update(params)
    if(mode =="Strong"):
        title_str = "Strong Students"
    elif(mode ==  "Weak"):
        title_str = "Weak Students"
    elif(mode=="Gaussian"):
        title_str="Data with Gaussian Distribution"
    else:
        title_str="Data with Uniform Distribution"

    
    algo_num = len(logger_list)
    legend_name = names
    
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
    plt.title(title_str)
    #plt.ylim(0,1)
    
    fig = plt.gcf()
    #fig.set_size_inches(18.5, 10.5)
    fig.savefig('pic/'+title_str+'.png', dpi=600)
    plt.show()

#          
        #####################  action quality

   
def plot_arr_pfmc(logger_list,legend_name,mode=None):
    if(mode =="Strong"):
        title_str = "Strong Students"
        
    elif(mode ==  "Weak"):
        title_str = "Weak Students"
    else:
        title_str=""
        
    algo_num = len(logger_list)

#    
    plt.figure(figsize=(5,4))
    ax=plt.subplot(111)
    
    for i in range(algo_num):
          sr = logger_list[i].arr_pfmc
          time_match = np.mean(sr,axis = 0)
          plt.plot(time_match,legends[i])
          #temp_name.append(names[i]+' For WS')
    if(mode ==  "Weak"):
        plt.legend(legend_name[:algo_num],loc='lower right', bbox_to_anchor=(1, 0.7),
          ncol=2, fancybox=True, shadow=True)
    else:
        plt.legend(legend_name[:algo_num],loc='lower right', bbox_to_anchor=(1, 0),
          ncol=2, fancybox=True, shadow=True)
    #plt.ylabel("Percieved Difficulty")
    plt.ylim(0,1)
    #plt.yticks([0, 0.5,1.0], ["",'',''])#
    plt.yticks([0, 0.5,1.0],["Hard", "Suitable","Easy"])
    plt.title(title_str)
    
    plt.xlabel("Time step")
    fig = plt.gcf()
    #fig.set_size_inches(18.5, 10.5)
    fig.savefig('pic/'+title_str+'.png', dpi=600)
  
    

    
