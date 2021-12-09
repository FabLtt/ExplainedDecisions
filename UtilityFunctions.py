# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:26:17 2020

@author: FabLtt

Scope: Utility functions to coumpute input features 
"""

import numpy as np
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

        
def find_first_time_active(Agent_list):
    
    agent_tot = len(Agent_list)
    time_instants_tot = len(Agent_list[0]['Position'])
    
    t_active_array = np.zeros(shape = (agent_tot,time_instants_tot))
    count = 0

    for agent_dict in Agent_list:
        t_active_array[count,:] = agent_dict['Active']
        count += 1

    first_time_active = np.min(np.where(np.sum(t_active_array, axis=0) == agent_tot))
    
    if(first_time_active == 0):
        first_time_active = np.min(np.where(np.sum(t_active_array[:,1:], axis=0) == agent_tot))
    
    return first_time_active

def find_first_time_contained(Agent_list, data):
    
    agent_tot = len(Agent_list)
    time_instants_tot = len(data['META']['TimeStamps'])
    
    t_contained_array = np.zeros(shape = (agent_tot,time_instants_tot))

    count_agent = 0

    for agent_dict in Agent_list:
        t_contained_array[count_agent,:] = agent_dict['Contained']
        count_agent += 1
        
    find_the_values = np.where(np.sum(t_contained_array, axis=0) == agent_tot)  
    first_time_contained = np.min(find_the_values)
    
    if(first_time_contained == 0):
        first_time_contained = np.min(np.where(np.sum(t_contained_array[:,1:], axis=0) == agent_tot))
    
    return first_time_contained

def get_agents_positions(Agent_list, time_range):

    agent_tot = len(Agent_list)
    coord_number = 2; 

    agent_UNcontained_positions = np.zeros(shape = (agent_tot, coord_number, time_range))

    count_targets = 0

    for agent_dict in Agent_list:
        position = agent_dict['Position']
        for time in range(time_range):  
            pos_at_t = position[time]
            pos_x = pos_at_t["x"]
            pos_z = pos_at_t["z"]
            agent_UNcontained_positions[count_targets,:,time] = [pos_x, pos_z]
        count_targets += 1
    
    return agent_UNcontained_positions

def distance_among_agents(herder_pos,target_pos,time_range):
    
    distance_from_herder = np.zeros(shape =(time_range, np.shape(target_pos)[0], np.shape(herder_pos)[0]))

    for time in range(time_range):
        for target in range(len(target_pos)):
            for herder in range(len(herder_pos)):
                distance_from_herder[time,target,herder] = np.linalg.norm(target_pos[target,:,time] - herder_pos[herder,:,time])
    return distance_from_herder

def integrate_in_time(data, agent_var, time_range):
    
    time_stamp = np.empty(len(data['META']['TimeStamps']))
    
    for i in range(len(data['META']['TimeStamps'])):
        time_stamp[i] = data['META']['TimeStamps'][i]
    
    delta_t = time_stamp[1:] - time_stamp[:-1]

    if agent_var.ndim == 3:
        agent_dot = np.empty(shape = (len(agent_var) , agent_var.shape[1] , time_range -1))
        for target in range(len(agent_var)):
            for coord in range(agent_var.shape[1]):
                agent_dot[target,coord,:] = (agent_var[target,coord,1:] - agent_var[target,coord,:-1])/delta_t[0:time_range-1]
    else: 
        agent_dot = np.empty(shape = (time_range -1, agent_var.shape[1]))
        for target in range(agent_var.shape[1]):
            agent_dot[:,target] = (agent_var[1:,target] - agent_var[:-1,target])/delta_t[0:time_range-1]
    
    return agent_dot

def integrate_in_time_norm(agent_var, time_range):
    
    agent_dot_norm = np.empty(shape=(time_range-1, len(agent_var)))
    
    for time in range(time_range - 1):
        for target in range(len(agent_var)):
            agent_dot_norm[time,target] = np.linalg.norm(agent_var[target,:,time])
    
    return agent_dot_norm

def get_direction_of_motion(agent_vel, agent_goal_pos):
    
    direction = np.empty(shape=(agent_vel.shape[2],agent_vel.shape[0]))
    
    for time in range(agent_vel.shape[2]):
        for target in range(agent_vel.shape[0]):
            direction[time, target] = np.dot(agent_vel[target,:,time], agent_goal_pos[target,:,time])
    
    return direction
    
def get_angular_velocity(agent_var):
    
    agent_var_phase = np.empty(shape=(agent_var.shape[2],agent_var.shape[0]))
    
    for time in range(agent_var.shape[2]):
        for target in range(agent_var.shape[0]):
            agent_var_phase[time,target] = math.atan2(agent_var[target,1,time], agent_var[target,0,time])
    
    return agent_var_phase

def distance_from_goal(agent_pos,goal_pos):
    
    distance = np.zeros(shape =(agent_pos.shape[2], agent_pos.shape[0]))

    for time in range(agent_pos.shape[2]):
        for target in range(agent_pos.shape[0]):
            distance[time,target] = np.linalg.norm(agent_pos[target,:,time] - goal_pos)
    return distance

def angle_from_goal(agent_pos,goal_pos):
    
    angle = np.zeros(shape =(agent_pos.shape[2], agent_pos.shape[0]))

    for time in range(agent_pos.shape[2]):
        for target in range(agent_pos.shape[0]):
            point = agent_pos[target,:,time] - goal_pos
            angle[time,target] =  math.atan2(point[0][1], point[0][0])
    return angle

def create_dataset(dataset, look_back=1, look_forward=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-look_forward):
        a = dataset[i:(i+look_back), :-1]
        dataX.append(a)
        dataY.append(dataset[i + look_back + look_forward, -1])
    return dataX, dataY 

def generate_model(dropout, neuronPct, neuronShrink,n_features,look_back):
    neuronCount = int(neuronPct * 2500)
    
    # Construct neural network
    model = Sequential()
    layer = 0
    
    while neuronCount>25 and layer<10:
        if layer==0:
            model.add(LSTM(neuronCount,input_shape=(look_back, n_features), return_sequences=True, dropout=dropout+0.1))
        else:
            model.add(LSTM(neuronCount,input_shape=(look_back, n_features), return_sequences=True, dropout=dropout+0.1)) 
        layer += 1

        # Add dropout after each LSTM layer
        model.add(Dropout(dropout,input_shape=(look_back, n_features)))
        neuronCount = int(neuronCount * neuronShrink)
        
    model.add(LSTM(neuronCount,input_shape=(look_back, n_features), dropout=dropout+0.1)) 
    model.add(Dense(5,activation='softmax')) # Output multi
    return model