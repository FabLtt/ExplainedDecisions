# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:14:57 2020

@author: FabLtt

Scope: cross-validate the ANN, of alternating LSTM and Dropout layers 

Note: code developed on tensorflow 1.15.0 and sklearn 0.23.1
"""
# Supress NaN warnings
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

from sklearn.model_selection import StratifiedShuffleSplit, KFold

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.callbacks import EarlyStopping


print("\n\n starting..")

# Choose expertise and delta time to train the ANN on
expertise = 'Novice' # options: 'Novice' | 'Expert'
step = 1


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

# Set if type of samples need to be balanced (True) or extracted with the same distribution as in the whole dataset (False)
balancedTypes = True

# Set training paramenters
learningRate_val = 0.0018
patience_val  = 15
epochs_val = 100

# Open the dataset file
file = open("./Datasets/DatasetFile_"+expertise+"_step"+str(step),"rb")


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
print("\n there are ", n_features," features!")

Dataset = Dataset_df.values

sequences, sequences_labels, targets = [],[],[]

herders_tot = int(max(Dataset[:,0])) + 1
trial_tot = int(max(Dataset[:,1])) + 1

for herder_id in range(herders_tot):
    for trial_id in range(trial_tot):
        Dtst = Dataset_df[(Dataset_df["Herder_id"]==herder_id) & (Dataset_df["Trial_id"]==trial_id)].values[:,2:]
        seq, tar, seq_lbl = uf.create_dataset(Dtst, look_back, look_forward)
        sequences = sequences + seq
        targets = targets + tar
        sequences_labels = sequences_labels + seq_lbl
        
sequences_array = np.array(sequences)
targets_array = np.array(targets)
sequences_labels_array = np.array(sequences_labels)

occurrences, count = np.unique(targets, return_counts=True)
n_classes = len(count)

all_indexes = []
for w_index in range(len(targets)):
    all_indexes.append(w_index)
all_indexes_array = np.array(all_indexes)
print('total number of samples:',len(all_indexes))

print('Distribution of samples by type on %i samples'%len(sequences_labels_array))
targets_labels_array = uf.checkSamplesType(sequences_labels_array)


if balancedTypes == True:
    # Split samples to balance type of samples 
    less_represented_class_size = 6000
    type_index = []
    for c in np.unique(targets_labels_array):
        indices = np.where(targets_labels_array == c)[0]
        sampled_indices = np.random.choice(indices, size=less_represented_class_size, replace=False)
        type_index.extend(sampled_indices)
else:   
    # Split samples to balance type of samples to reflect their distribution in the whole dataset
    sss = StratifiedShuffleSplit(train_size=len(targets_array)-n_classes, n_splits=1, 
                                  test_size=n_classes, random_state=0)  
    
    
X = sequences_array[type_index]
y = targets_array[type_index]
Z = sequences_labels_array[type_index]

print('Distribution of samples by type, after splitting')
temp = uf.checkSamplesType(Z)
del temp    


    
# Check if the size of training and test set overflow the total available number of samples
if (test_samples+train_samples)>X.shape[0]:
    train_samples = X.shape[0] - 2000
    print("max samples reached (",train_samples,")!")
   

# Randomly select 'train_samples' and 'test_samples' samples from the total available ones (i.e. 'train' and 'train_target')
sss = StratifiedShuffleSplit(train_size=train_samples, n_splits=1, 
                             test_size=test_samples, random_state=0)  

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    Z_train, Z_test = Z[train_index], Z[test_index]
    
del X, y, Z
    
    
print("training samples :", y_train.shape)
print("testing samples :", y_test.shape)
# occurrences, count = np.unique(y_train, return_counts=True)
# print(count/len(y_train))

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

# Cross validation

num_folds = 5

epochs_needed = []
acc_per_fold = []
loss_per_fold = []


# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# Define early stopping condition
earlStop = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=patience_val, restore_best_weights=True, verbose = 0)

# K-fold Cross Validation model evaluation
fold_no = 1
for train_n, val_n in kfold.split(X, y):

    model = uf.generate_model(dropout=0.1145, neuronPct=0.1012, neuronShrink=0.1793, n_features = n_features,look_back=look_back) # model H

    # Compile the model
    adam = Adam(lr=learningRate_val) 

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
    
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history = model.fit(X[train_n], y[train_n], validation_data=[X[val_n], y[val_n]], batch_size=32, epochs=epochs_val, verbose=0, callbacks=[earlStop])
    epochs = earlStop.stopped_epoch
    epochs_needed.append(epochs)
    
    save_model(model, checkpoint_file+'_'+str(fold_no), overwrite=True, include_optimizer=True, save_format=None, signatures=None,)

    # Generate generalization metrics on validation dataset
    scores = model.evaluate(X[val_n], y[val_n], verbose=0)
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1
    

# Average scores on test set 
acc_per_fold_test = []
loss_per_fold_test = []

print('------------------------------------------------------------------------')
print('Score per fold on test set ')
for i in range(1,num_folds+1):
    model = load_model(checkpoint_file+'_'+str(i))
    scores = model.evaluate(test, test_target, verbose=0)
    acc_per_fold_test.append(scores[1]*100)
    loss_per_fold_test.append(scores[0])
    print("Fold ",i,"---- Accuracy: %.2f%%" % (scores[1]*100))
    
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold_test)} (+- {np.std(acc_per_fold_test)})')
print(f'> Loss: {np.mean(loss_per_fold_test)}')