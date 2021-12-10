# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:14:57 2020

@author: FabLtt

Scope: train the ANN, of alternating LSTM and Dropout layers 

Note: code developed on tensorflow 1.15.0 and sklearn 0.23.1
"""

import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pickle   
import datetime
import numpy as np
import pandas as pd 

import UtilityFunctions as uf

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf


print("\n\n starting on tf version ", tf.__version__)


# Choose expertise and delta time to train the ANN on
expertise = 'Novice' # options: 'Novice' | 'Expert'
step = 1

# Open the dataset file
file = open("./Datasets/DatasetFile_"+expertise+"_step"+str(step),"rb")  

# Set the name to be saved for the model that will be trained
file_id = '001'
checkpoint_file = "./checkpoint/Model"+expertise+"Step"+str(step)+"_"+ datetime.datetime.now().strftime("%d%m%Y")+"_"+file_id                  
file_name_index = "./checkpoint/Model"+expertise+"Step"+str(step)+"_"+ datetime.datetime.now().strftime("%d%m%Y")+"_"+file_id +"_TrainTestSets"   # Where indexes of training and test samples are saved


# Set sequence length 'look_back' and decision horizon 'look_forward'= [1, 8, 16, 32]
look_back = 25
look_forward = 1 

# Set training set and test set size
train_samples = 21000
test_samples = 2000


# Load the dataset and select the columns referred by 'Labels'
Dataset_full_df = pickle.load(file)
file.close()

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
print("\n there are ", n_features," features")

Dataset = Dataset_df.values

sequences, targets = [],[]

herders_tot = int(max(Dataset[:,0]))+1
trial_tot = int(max(Dataset[:,1]))+1

# A sequence can refer to only one Herder ID and one Trial ID
for herder_id in range(herders_tot): 
    for trial_id in range(trial_tot):
        Dtst = Dataset_df[(Dataset_df["Herder_id"]==herder_id) & (Dataset_df["Trial_id"]==trial_id)].values[:,2:]
        seq, tar = uf.create_dataset(Dtst, look_back, look_forward)
        sequences = sequences + seq
        targets = targets + tar

train = np.array(sequences)
train_target = np.array(targets)

# Check if the size of training and test set overflow the total available number of samples
if (test_samples+train_samples)>train.shape[0]:
    train_samples = train.shape[0] - 2000
    print("max samples reached (",train_samples,")!")

# Randomly select 'train_samples' and 'test_samples' samples from the total available ones (i.e. 'train' and 'train_target')
sss = StratifiedShuffleSplit(train_size=train_samples, n_splits=1, 
                             test_size=test_samples, random_state=0)  

for train_index, test_index in sss.split(train, train_target):
    X_train, X_test = train[train_index], train[test_index]
    y_train, y_test = train_target[train_index], train_target[test_index]
    
    
print("training samples :", y_train.shape)
print("testing samples :", y_test.shape)


# Save the indexes of training and test samples
file_to_save = open(file_name_index,"wb")
data_to_save = [train_index, test_index]  
pickle.dump(data_to_save,file_to_save)
file_to_save.close()


# Rename training and test samples to be fed the ANN
dummies_train = pd.get_dummies(y_train)

X = X_train
y = dummies_train.values

dummies_test = pd.get_dummies(y_test)

test = X_test
test_target = dummies_test.values


model = uf.generate_model(dropout=0.1145, neuronPct=0.1012, neuronShrink=0.1793, n_features = n_features, look_back = look_back) # model H

# Compile the ANN model
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

earlStop = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, restore_best_weights=True, verbose = 1)

adam = Adam(lr=0.0018) 

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])

# Fit data to ANN model
history = model.fit(X_train, y_train, validation_data=[X_val, y_val], batch_size=32, epochs=100, verbose=0,callbacks=[earlStop])

epochs = earlStop.stopped_epoch

# Save the trained ANN model
save_model(model, checkpoint_file, overwrite=True, include_optimizer=True, save_format=None, signatures=None,)


# Provide average scores on test set 
scores = model.evaluate(test, test_target, verbose=0)
print("---- Loss: %.2f" % (scores[0]))
print("---- Accuracy: %.2f%%" % (scores[1]*100))





