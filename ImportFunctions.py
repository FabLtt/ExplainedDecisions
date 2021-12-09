# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:26:17 2020

@author: FabLtt

Scope: Utility functions to import features into a unified dataset
"""

import json
import os
import pickle 
import math  

import numpy as np


def ImportDatasetFromJsonFileFull_MultiPredWrtGoal(directory, herder, starting_string="LABELED_", both=False, coding=True, alltime=False, after=False, both_predict=False, extended=False):
    
    N_features = 43 + 30
    
    if both == True:
        N_features = 43 + 30 + 12
        
        if extended==True:
            N_features = 43 + 30 + 12 + 8

    trial_ID = int(0)
              
    for entry in os.scandir(directory):
        if(entry.name.startswith(starting_string)) and entry.is_file():
    #         print("\n", entry.name)
            with open(entry) as f:
                data_train = json.load(f)
    
                file_name_targets = directory + '/' + data_train['META']['ReferenceID'] + data_train['META']['FileName']+'_features'
                file_name_herders = directory + '/' + data_train['META']['ReferenceID'] + data_train['META']['FileName']+'_features_herders'
                
                if alltime == True:
                    file_name_targets = directory + '/' + data_train['META']['ReferenceID'] + data_train['META']['FileName']+'_features_ALLTIME'
                    file_name_herders = directory + '/' + data_train['META']['ReferenceID'] + data_train['META']['FileName']+'_features_herders_ALLTIME'
                 
                with open(file_name_targets,'rb') as f:
                    features_targets = pickle.load(f)
    
                with open(file_name_herders,'rb') as f:
                    features_herders = pickle.load(f)
    
                N_timesteps = features_targets['FirstTimeContained'] - features_targets['FirstTimeActive'] -2
                
                start_time = 0
                
                # print(N_timesteps)
                
                if after == True:
                    
                    Agent_list = data_train['Targets'] 
                    agent_tot = len(Agent_list)
                    time_instants_tot = len(data_train['META']['TimeStamps'])
                    
                    t_contained_array = np.zeros(shape = (agent_tot,time_instants_tot))
                
                    count_agent = 0
                
                    for agent_dict in Agent_list:
                        t_contained_array[count_agent,:] = agent_dict['Contained'][:time_instants_tot]
                        count_agent += 1
                
                    start_time = np.min(np.where(np.sum(t_contained_array, axis=0) == agent_tot))
                    
                    if(start_time == 0):
                        start_time = np.min(np.where(np.sum(t_contained_array[:,1:], axis=0) == agent_tot))
                       
                    N_timesteps = features_targets['FirstTimeContained'] - features_targets['FirstTimeActive'] -2 - start_time
                
                # print(N_timesteps)
                goal_centre_pos = features_targets["GoalPos"]
                target_pos = features_targets["TargetPos"]
                herder_pos = features_targets["HerderPos"]
                
                
                distance_among_agents = features_targets["DistanceFromHerders"] #time x agents 
                
                angle_among_agents = features_targets["AngleFromHerders"]
                
                distance_from_goal_herders = features_herders["DistanceFromGoal"]
                distance_from_goal_targets = features_targets["DistanceFromGoal"]
                              
                velocity_norm_herders = features_herders["HerderVelNorm"] #time x agents 
                velocity_phase_herders = features_herders["HerderVelPhase"]
                velocity_norm_targets = features_targets["TargetVelNorm"]
                velocity_phase_targets = features_targets["TargetVelPhase"]
                
                acceleration_norm_herders = features_herders["HerderAccNorm"] #time x agents 
                acceleration_norm_targets = features_targets["TargetAccNorm"]
                
                #################################################
                angle_from_goal_targets = features_targets["AngleFromGoal"]  #time x agents 
                dir_motion_targets = features_targets["TargetDirMotion"] #
                velocity_wrtgoal_norm_targets = features_targets["TargetWRTGoalVelNorm"]
                velocity_wrtgoal_phase_targets = features_targets["TargetWRTGoalVelPhase"]
                acceleration_wrtgoal_norm_targets = features_targets["TargetWRTGoalAccNorm"]
                acceleration_wrtgoal_phase_targets = features_targets["TargetWRTGoalAccPhase"]
                
                angle_from_goal_herders = features_herders["AngleFromGoal"]  #time x agents 
                dir_motion_herders = features_herders["HerderDirMotion"] #
                velocity_wrtgoal_norm_herders = features_herders["HerderWRTGoalVelNorm"]
                velocity_wrtgoal_phase_herders = features_herders["HerderWRTGoalVelPhase"]
                acceleration_wrtgoal_norm_herders = features_herders["HerderWRTGoalAccNorm"]
                acceleration_wrtgoal_phase_herders = features_herders["HerderWRTGoalAccPhase"]
                               
                Herders_Dataset_train = np.empty(shape=(N_timesteps,N_features))
 
                
                for time in range(N_timesteps):   
                    

                        t = start_time + time
                        
                        #trial_IDs
                        Herders_Dataset_train[time,0] = herder
                        Herders_Dataset_train[time,1] = trial_ID
                    
                        #features
                        Herders_Dataset_train[time,2] = goal_centre_pos[0,0]
                        Herders_Dataset_train[time,3] = goal_centre_pos[0,1]
                        
                        Herders_Dataset_train[time,4] = target_pos[0,0,t] # target 0 pos x 
                        Herders_Dataset_train[time,5] = target_pos[0,1,t] # target 0 pos y
                        Herders_Dataset_train[time,6] = target_pos[1,0,t] # target 1 pos x 
                        Herders_Dataset_train[time,7] = target_pos[1,1,t] # target 1 pos y 
                        Herders_Dataset_train[time,8] = target_pos[2,0,t] # target 2 pos x 
                        Herders_Dataset_train[time,9] = target_pos[2,1,t] # target 2 pos y 
                        Herders_Dataset_train[time,10] = target_pos[3,0,t] # target 3 pos x 
                        Herders_Dataset_train[time,11] = target_pos[3,1,t] # target 3 pos y 
                        
                        Herders_Dataset_train[time,12] = herder_pos[herder,0,t] # herder pos x
                        Herders_Dataset_train[time,13] = herder_pos[herder,1,t] # herder pos y
                            
                        Herders_Dataset_train[time,14] = distance_among_agents[t,0,herder] #0
                        Herders_Dataset_train[time,15] = distance_among_agents[t,1,herder] #0
                        Herders_Dataset_train[time,16] = distance_among_agents[t,2,herder] #0
                        Herders_Dataset_train[time,17] = distance_among_agents[t,3,herder] #0
                        
                        Herders_Dataset_train[time,18] = angle_among_agents[t,0,herder] #1
                        Herders_Dataset_train[time,19] = angle_among_agents[t,1,herder] #1
                        Herders_Dataset_train[time,20] = angle_among_agents[t,2,herder] #1
                        Herders_Dataset_train[time,21] = angle_among_agents[t,3,herder] #1
                        

                        Herders_Dataset_train[time,22] = distance_from_goal_herders[t,herder] #3
                        Herders_Dataset_train[time,23] = distance_from_goal_targets[t,0] #4
                        Herders_Dataset_train[time,24] = distance_from_goal_targets[t,1] #4
                        Herders_Dataset_train[time,25] = distance_from_goal_targets[t,2] #4
                        Herders_Dataset_train[time,26] = distance_from_goal_targets[t,3] #4

                        Herders_Dataset_train[time,27] = velocity_norm_herders[t,herder] #11
                        Herders_Dataset_train[time,28] = velocity_phase_herders[t,herder] #12
                        Herders_Dataset_train[time,29] = velocity_norm_targets[t,0]  #13
                        Herders_Dataset_train[time,30] = velocity_phase_targets[t,0] #14
                        Herders_Dataset_train[time,31] = velocity_norm_targets[t,1]  #13
                        Herders_Dataset_train[time,32] = velocity_phase_targets[t,1] #14
                        Herders_Dataset_train[time,33] = velocity_norm_targets[t,2]  #13
                        Herders_Dataset_train[time,34] = velocity_phase_targets[t,2] #14
                        Herders_Dataset_train[time,35] = velocity_norm_targets[t,3]  #13
                        Herders_Dataset_train[time,36] = velocity_phase_targets[t,3] #14
                        
                        Herders_Dataset_train[time,37] = acceleration_norm_herders[t,herder] #19
                        Herders_Dataset_train[time,38] = acceleration_norm_targets[t,0] #20
                        Herders_Dataset_train[time,39] = acceleration_norm_targets[t,1] #20
                        Herders_Dataset_train[time,40] = acceleration_norm_targets[t,2] #20
                        Herders_Dataset_train[time,41] = acceleration_norm_targets[t,3] #20
                        
                        Herders_Dataset_train[time,42] = angle_from_goal_herders[t,herder] 
                        Herders_Dataset_train[time,43] = angle_from_goal_targets[t,0]
                        Herders_Dataset_train[time,44] = angle_from_goal_targets[t,1]
                        Herders_Dataset_train[time,45] = angle_from_goal_targets[t,2]
                        Herders_Dataset_train[time,46] = angle_from_goal_targets[t,3]
                        
                        Herders_Dataset_train[time,47] = dir_motion_herders[t,herder]
                        Herders_Dataset_train[time,48] = dir_motion_targets[t,0]
                        Herders_Dataset_train[time,49] = dir_motion_targets[t,1]
                        Herders_Dataset_train[time,50] = dir_motion_targets[t,2]
                        Herders_Dataset_train[time,51] = dir_motion_targets[t,3]
                        
                        Herders_Dataset_train[time,52] = velocity_wrtgoal_norm_herders[t,herder]
                        Herders_Dataset_train[time,53] = velocity_wrtgoal_phase_herders[t,herder]
                        
                        Herders_Dataset_train[time,54] = velocity_wrtgoal_norm_targets[t,0]
                        Herders_Dataset_train[time,55] = velocity_wrtgoal_phase_targets[t,0]
                        Herders_Dataset_train[time,56] = velocity_wrtgoal_norm_targets[t,1]
                        Herders_Dataset_train[time,57] = velocity_wrtgoal_phase_targets[t,1]
                        Herders_Dataset_train[time,58] = velocity_wrtgoal_norm_targets[t,2]
                        Herders_Dataset_train[time,59] = velocity_wrtgoal_phase_targets[t,2]
                        Herders_Dataset_train[time,60] = velocity_wrtgoal_norm_targets[t,3]
                        Herders_Dataset_train[time,61] = velocity_wrtgoal_phase_targets[t,3]
                        
                        Herders_Dataset_train[time,62] = acceleration_wrtgoal_norm_herders[t,herder]
                        Herders_Dataset_train[time,63] = acceleration_wrtgoal_phase_herders[t,herder]
                        
                        Herders_Dataset_train[time,64] = acceleration_wrtgoal_norm_targets[t,0]
                        Herders_Dataset_train[time,65] = acceleration_wrtgoal_phase_targets[t,0]
                        Herders_Dataset_train[time,66] = acceleration_wrtgoal_norm_targets[t,1]
                        Herders_Dataset_train[time,67] = acceleration_wrtgoal_phase_targets[t,1]
                        Herders_Dataset_train[time,68] = acceleration_wrtgoal_norm_targets[t,2]
                        Herders_Dataset_train[time,69] = acceleration_wrtgoal_phase_targets[t,2]
                        Herders_Dataset_train[time,70] = acceleration_wrtgoal_norm_targets[t,3]
                        Herders_Dataset_train[time,71] = acceleration_wrtgoal_phase_targets[t,3]
        
                        #label
                        if coding == True:
                            Herders_Dataset_train[time,72] = data_train['Herders'][herder]['TargetedTarget'][t]
                        
                        if both == True: 
                            if herder == 0:
                                other_herder = 1
                            else:
                                other_herder = 0
                            
                            ## print(herder, other_herder)
                            
                            rel_pos = herder_pos[herder,:,time] - herder_pos[other_herder,:,t]
                            Herders_Dataset_train[time,72] = np.linalg.norm(rel_pos)  # radial distance herder from other_herder
                            Herders_Dataset_train[time,73] = math.atan2(rel_pos[1],rel_pos[0]) # angle herder from other_herder
                            Herders_Dataset_train[time,74] = distance_from_goal_herders[t,other_herder] # radial distance from goal 
                            Herders_Dataset_train[time,75] = velocity_norm_herders[t,other_herder] # radial velocity other_herder
                            Herders_Dataset_train[time,76] = velocity_phase_herders[t,other_herder] # angular velocity other_herder
                            Herders_Dataset_train[time,77] = acceleration_norm_herders[t,other_herder] # radial acceleration other_herder
                            
                            Herders_Dataset_train[time,78] = angle_from_goal_herders[t,other_herder]
                            Herders_Dataset_train[time,79] = dir_motion_herders[t,other_herder]
                            
                            Herders_Dataset_train[time,80] = velocity_wrtgoal_norm_herders[t,other_herder]
                            Herders_Dataset_train[time,81] = velocity_wrtgoal_phase_herders[t,other_herder]
                            
                            Herders_Dataset_train[time,82] = acceleration_wrtgoal_norm_herders[t,other_herder]
                            Herders_Dataset_train[time,83] = acceleration_wrtgoal_phase_herders[t,other_herder]
                            
                            if extended==True:
                                last_index = 92
                                Herders_Dataset_train[time,84] = distance_among_agents[t,0,other_herder] #0
                                Herders_Dataset_train[time,85] = distance_among_agents[t,1,other_herder] #0
                                Herders_Dataset_train[time,86] = distance_among_agents[t,2,other_herder] #0
                                Herders_Dataset_train[time,87] = distance_among_agents[t,3,other_herder] #0
                                
                                Herders_Dataset_train[time,88] = angle_among_agents[t,0,other_herder] #1
                                Herders_Dataset_train[time,89] = angle_among_agents[t,1,other_herder] #1
                                Herders_Dataset_train[time,90] = angle_among_agents[t,2,other_herder] #1
                                Herders_Dataset_train[time,91] = angle_among_agents[t,3,other_herder] #1
                                
                            else:
                                last_index = 84
   
                            # (new) label
                            if coding == True:
                                
                                if both_predict == True:
                                    
                                    label1 = data_train['Herders'][herder]['TargetedTarget'][t]
                                    label2 = data_train['Herders'][other_herder]['TargetedTarget'][t]
                            
                                    Herders_Dataset_train[time,last_index] = str(label1) + str(label2)
                                    
                                else:
                                
                                    Herders_Dataset_train[time,last_index] = data_train['Herders'][herder]['TargetedTarget'][t]
                                
                                
                                
                
                Herders_Dataset_train[:,-1] = Herders_Dataset_train[:,-1].astype(int)
                
                if trial_ID == 0:
                    Dataset_ = Herders_Dataset_train
                else:
                    Dataset_ = np.append(Dataset_,Herders_Dataset_train,axis=0)
                
                trial_ID += 1
                
    print("appended to the dataset up to ", Dataset_.shape[0]," elements")

    return Dataset_ 



def giveLabels(keyword):
    
    if keyword == "full_ts_multi_both_wrtgoal_extended":
        features_full4_timeseries_multi = ['goal posX', 'goal posY', 'target0 posX', 'target0 posY','target1 posX', 'target1 posY', 'target2 posX', 'target2 posY', 'target3 posX', 'target3 posY', 'herder posX','herder posY',
                                          'h_t0 rel dist', 'h_t1 rel dist', 'h_t2 rel dist', 'h_t3 rel dist', 'h_t0 rel angle', 'h_t1 rel angle', 'h_t2 rel angle', 'h_t3 rel angle', 
                                          'h_goal rel dist', 't0_goal rel dist', 't1_goal rel dist', 't2_goal rel dist', 't3_goal rel dist', 
                                          'h vel_r' , 'h vel_th', 't0 vel_r' , 't0 vel_th', 't1 vel_r' , 't1 vel_th', 't2 vel_r' , 't2 vel_th', 't3 vel_r' , 't3 vel_th',
                                          'h acc_r', 't0 acc_r', 't1 acc_r', 't2 acc_r', 't3 acc_r', 'h_goal_th', 't0_goal_th', 't1_goal_th', 't2_goal_th', 't3_goal_th',
                                          'h_dir_motion', 't0_dir_motion', 't1_dir_motion', 't2_dir_motion', 't3_dir_motion',
                                          'h velwrtgoal_r' , 'h velwrtgoal_th', 't0 velwrtgoal_r' , 't0 velwrtgoal_th', 't1 velwrtgoal_r' , 't1 velwrtgoal_th', 't2 velwrtgoal_r' , 
                                          't2 velwrtgoal_th', 't3 velwrtgoal_r' , 't3 velwrtgoal_th', 'h accwrtgoal_r' , 'h accwrtgoal_th', 't0 accwrtgoal_r' , 't0 accwrtgoal_th', 
                                          't1 accwrtgoal_r' , 't1 accwrtgoal_th', 't2 accwrtgoal_r' , 't2 accwrtgoal_th', 't3 accwrtgoal_r' , 't3 accwrtgoal_th', 
                                          'h_h1 rel dist', 'h_h1 rel angle', 'h1_goal rel dist', 'h1 vel_r', 'h1 vel_th','h1 acc_r',
                                          'h1_goal_th', 'h1_dir_motion', 'h1 velwrtgoal_r' , 'h1 velwrtgoal_th', 'h1 accwrtgoal_r' , 'h1 accwrtgoal_th',
                                          'h1_t0 rel dist', 'h1_t1 rel dist', 'h1_t2 rel dist', 'h1_t3 rel dist', 'h1_t0 rel angle', 'h1_t1 rel angle', 'h1_t2 rel angle', 'h1_t3 rel angle'
                                          ]
        features = features_full4_timeseries_multi
    
        
    features.append('Label')
   
             
    return features
        