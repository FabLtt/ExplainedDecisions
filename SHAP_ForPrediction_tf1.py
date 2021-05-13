# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 17:58:17 2020

@author: fa17936

Scope: compute SHAP values 
"""
import pickle   
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import shap
import pandas as pd

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Select the number of samples from the training set to use as background and the number of samples to explain 
n_background = 1000
n_samples = 2000

# Select the decision horizon [1, 8, 16, 32]
look_forward = 1

# Notice that DeepExplainer is not compatible with TF2
print("\n\n starting on tf version ", tf.__version__, "and shap version", shap.__version__)

# Load the dataset, the trained model and training and test indexes' files
file = open("./dataset/DatasetFileMultiClassPred_BothHerders_WrtGoal_Extended","rb")

directory = "./checkpoint/"
file_model_name ="Model_13052021_001"
file_to_open = open(directory+'TrainTestSets_indexes_001',"rb")

Dataset_full_df = pickle.load(file)


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

# Create the sequences of features and target outputs from the datas
def create_dataset(dataset, look_back=1, look_forward=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-look_forward):
        a = dataset[i:(i+look_back), :-1]
        dataX.append(a)
        dataY.append(dataset[i + look_back + look_forward, -1])
    return dataX, dataY 

look_back = 25   # Sequence length
sequences = []
targets = []

herders_tot = int(max(Dataset[:,0]))
trial_tot = int(max(Dataset[:,1]))

# A sequence can refer to only one Herder ID and one Trial ID
for herder_id in range(herders_tot):
    for trial_id in range(trial_tot):
        Dtst = Dataset_df[(Dataset_df["Herder_id"]==herder_id) & (Dataset_df["Trial_id"]==trial_id)].values[:,2:]
        seq, tar = create_dataset(Dtst, look_back, look_forward)
        sequences = sequences + seq
        targets = targets + tar

# Load the training and test samples 
indexes_data = pickle.load(file_to_open)
file_to_open.close()

train_index = indexes_data[0]
test_index = indexes_data[1]

sequences_array = np.array(sequences)
targets_array = np.array(targets)

X_train, X_test = sequences_array[train_index], sequences_array[test_index]
y_train, y_test = targets_array[train_index], targets_array[test_index]
  
dummies_train = pd.get_dummies(y_train)

train = X_train
train_target = dummies_train.values    
    
dummies_test = pd.get_dummies(y_test)

test = X_test
test_target = dummies_test.values

# Load the model
model = load_model(directory + file_model_name)

# Select the background samples and the test samples to explain
background = train[np.random.choice(train.shape[0], n_background, replace=False)]
samples = test #[np.random.choice(test.shape[0], n_samples, replace=False)]

# Explain the model's predictions using SHAP
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(samples)

# Print the mean prediction outputs 
print(explainer.expected_value)

# Save the SHAP values 
file_to_write = directory + file_model_name + '_ShapVal'

with open(file_to_write,"wb") as f:
    pickle.dump([shap_values],f)
