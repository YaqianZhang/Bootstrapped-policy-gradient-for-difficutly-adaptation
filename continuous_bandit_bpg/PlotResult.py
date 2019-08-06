#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 15:11:01 2018

@author: zhangyaqian
"""
import numpy as np
import matplotlib.pyplot as plt
import time

        
def plotTrack(paths_algos,field_name,names,update_steps):
    legends=['r<','b.','-*','-<','-^','-v','-.*','-o','.-','*-','<-','-o','.-']
    skip_idx=np.linspace(0,update_steps-1,update_steps/10-1);
    skip_idx=skip_idx.astype(int)
   # plt.figure()
    #update_steps=100#np.array(m_list[0]).shape[0]
    for i in range(len(paths_algos)) : 
        #print( len(mlist[i][0]))
        arr=np.array(paths_algos[i][field_name])
        #arr = np.reshape(np.array(mlist[i]),[update_steps,-1])
        #plt.plot(np.array(m_list[i])[:update_steps,0],np.array(m_list[i])[:update_steps,1],legends[i])
             
            
    
        plt.plot(arr[:update_steps,0],arr[:update_steps,1],legends[i])
    
    plt.legend(names)
#    plt.xlim([-100,1000])
#    plt.ylim([-100,800])
    plt.ylabel("Mean on 2nd dim")
    plt.xlabel("Mean on 1st dim")
    plt.title("#updates: "+str(update_steps))
              

def showResult(batch_size,action_dim,paths_algos,names):
    legends=['r-<','b.-','g-*','-<','-^','-v','-.*','-o','.-','*-','<-','-o','.-']
    

    algo_num =  len(paths_algos)
    iters = len(paths_algos[0]['reward_list'])
    epochs = int(iters/(10*action_dim))
 
            
    x_axis=np.linspace(1,iters-1,epochs-1) 
    x_axis = x_axis.astype(int)

    plt.figure()
    for i in range(algo_num) :          
            
        plt.plot(np.array(range(iters))[x_axis],np.array(paths_algos[i]['reward_list'])[x_axis],legends[i])
    
    plt.legend(names)
    plt.ylabel("Reward")
    plt.xlabel("Time Steps")
    plt.title(str(action_dim)+" action dimensions")
    
    
    plt.figure(figsize=(5,4))
    for i in range(algo_num) :  
        aa=np.abs(paths_algos[i]['reward_list'] )          
            
        plt.plot(np.log(x_axis)/np.log(10),np.log(aa)[x_axis]/np.log(10),legends[i])
    
    plt.legend(names)
    plt.title(str(action_dim)+" action dimensions")
    #plt.title("Target action at 40")
    plt.ylabel("log(Cost)")
    plt.xlabel("log(Time Steps)")
    fig = plt.gcf()
    #fig.set_size_inches(18.5, 10.5)
    fig.savefig('pic/logcost'+str(action_dim)+'.png', dpi=600)
    
#    update_range=[int(iters/5), iters]
#    
#   
#    
#    
#    for update_steps in update_range:
#        plt.figure()
#        plotTrack(paths_algos,'mean_list',names,update_steps)
#        
#        #plt.title("Mean #updates"+ str(update_steps))
#        plt.title("Mean #epoch:"+ str(update_steps/batch_size))
#        
#
#    for update_steps in update_range:
#        plt.figure()
#        plotTrack(paths_algos,'action_list',names,update_steps)
#        #plt.title("Action value #updates"+ str(update_steps))
#        plt.title("Action value #epoch:"+ str(update_steps/batch_size))
        
        

    

     