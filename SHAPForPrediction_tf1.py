# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 17:58:17 2020

@author: FabLtt

Scope: compute SHAP values 

Note: code developed on tensorflow 1.15.0 and sklearn 0.23.1
"""
import pickle   
import numpy as np

import UtilityFunctions as uf

from tensorflow.keras.models import load_model
import tensorflow as tf
import shap

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Select model identifiers 
model_id = '26022022'
file_id = '001'

# Select the expertise and sampling time 'step' 
expertise = 'Novice' # options: 'Novice' | 'Expert'
key_expertise = 0   # options: 0 = 'Novice'  | 1 = 'Expert'
step = 2 # options: 1 | 2 | 4 

# Select the number of samples from the training set to use as background and the number of samples to explain 
n_background = 200
n_samples = 6000

# Select the sequence length 'look_back' and decision horizon 'look_forward' [1, 8, 16, 32]
look_back = 25   # Sequence length
look_forward = 16

# Notice that DeepExplainer is not compatible with TF2
print("\n\n starting on tf version ", tf.__version__, "and shap version", shap.__version__)

# Load the dataset, the trained model and training and test indexes' files
file = open("./Datasets/DatasetFile_"+expertise+"_step"+str(step),"rb")  

directory = "./checkpoint/"
file_model_name = directory+"Model"+expertise+"Step"+str(step)+"_"+model_id+"_"+file_id                  

if key_expertise == 0:
    file_name_index = "TrainTestSets_Novice_step"+str(step)+"_thor"+str(look_forward)
else:
    file_name_index = "TrainTestSets_Expert_step"+str(step)+"_thor"+str(look_forward)

file_to_open = open(file_name_index,"rb")

Dataset_full_df = pickle.load(file)
file.close()

# From the dataset, selected the following columns (input feature set)
# Notice that they must coincide with the ones used in the training phase 
Labels = ['h_t0 rel dist', 'h_t1 rel dist', 'h_t2 rel dist', 'h_t3 rel dist', 'h_t0 rel angle', 'h_t1 rel angle', 'h_t2 rel angle', 'h_t3 rel angle', 
          'h_goal rel dist', 't0_goal rel dist', 't1_goal rel dist', 't2_goal rel dist', 't3_goal rel dist',
          'h vel_r' , 't0 vel_r' , 't1 vel_r' , 't2 vel_r' ,  't3 vel_r' , 
          'h acc_r', 't0 acc_r', 't1 acc_r', 't2 acc_r', 't3 acc_r', 
          'h_goal_th', 't0_goal_th', 't1_goal_th', 't2_goal_th', 't3_goal_th', 
          'h_dir_motion', 't0_dir_motion', 't1_dir_motion', 't2_dir_motion', 't3_dir_motion',
          'h_h1 rel dist', 'h_h1 rel angle', 'h1_goal rel dist', 'h1 vel_r', 'h1 acc_r',
          'h1_goal_th', 'h1_dir_motion', 'h1_t0 rel dist', 'h1_t1 rel dist', 'h1_t2 rel dist', 'h1_t3 rel dist', 
          'h1_t0 rel angle', 'h1_t1 rel angle', 'h1_t2 rel angle', 'h1_t3 rel angle','Label']

Labels.insert(0,"Herder_id")
Labels.insert(1,"Trial_id")

Dataset_df = Dataset_full_df[Labels]
n_features = len(Dataset_df.columns) - 3
print("\n there are ", n_features," features!")

Dataset = Dataset_df.values

sequences = []

herders_tot = int(max(Dataset[:,0])) + 1
trial_tot = int(max(Dataset[:,1])) + 1

for herder_id in range(herders_tot):
    for trial_id in range(trial_tot):
        Dtst = Dataset_df[(Dataset_df["Herder_id"]==herder_id) & (Dataset_df["Trial_id"]==trial_id)].values[:,2:]
        seq, tar, seq_lbl = uf.create_dataset(Dtst, look_back, look_forward)
        sequences = sequences + seq
        
        
sequences_array = np.array(sequences)

# Load the training and test samples 
indexes_data = pickle.load(file_to_open)
file_to_open.close()

type_index = indexes_data[0]
train_index = indexes_data[1]
test_index = indexes_data[2]

X_senior = sequences_array[type_index]
X_train, X_test = X_senior[train_index], X_senior[test_index]


# Load the model
model = load_model(directory + file_model_name)

# Select the background samples and the test samples to explain
background = X_train[np.random.choice(X_train.shape[0], n_background, replace=False)]
samples = X_test [np.random.choice(X_test.shape[0], n_samples, replace=False)]

# Explain the model's predictions using SHAP
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(samples)

# Print the mean prediction outputs 
print(explainer.expected_value)

# Save the SHAP values 
file_to_write = directory + file_model_name + '_ShapVal'

with open(file_to_write,"wb") as f:
    pickle.dump([shap_values],f)
