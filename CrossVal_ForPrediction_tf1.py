# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:14:57 2020

@author: fa17936

Scope: cross-validate the ANN, of alternating LSTM and Dropout layers 
"""
import pickle   

import numpy as np
import pandas as pd 


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.callbacks import EarlyStopping


print("\n\n starting..")

# Open the dataset file
file = open("./dataset/DatasetFileMultiClassPred_BothHerders_WrtGoal_Extended","rb")   


checkpoint_file = "./checkpoint/Model_13052021_001"                  # Where the trained model is saved


# Set sequence length 'look_back' and decision horizon 'look_forward'= [1, 8, 16, 32]
look_back = 25
look_forward = 1 

# Set training set and test set size
train_samples = 21000
test_samples = 2000

# Load the dataset and select the columns referred by 'Labels'
Dataset_full_df = pickle.load(file)

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

# Create the sequences of features and target outputs from the dataset
def create_dataset(dataset, look_back=1, look_forward=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-look_forward):
        a = dataset[i:(i+look_back), :-1]
        dataX.append(a)
        dataY.append(dataset[i + look_back + look_forward, -1])
    return dataX, dataY 

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

# Randomly select 'train_samples' and 'test_samples' samples from the total available ones (i.e. 'train' and 'train_target') 
tr, tst, tr_target, tst_target = train_test_split(sequences, targets, test_size=0.2, random_state=4)

dummies = pd.get_dummies(targets)
targets_dummies = dummies.values

train = [sequences[i] for i in range(0,len(tr))]
test = [sequences[i] for i in range(len(tr),len(sequences))]

train_target = [targets_dummies[i,:] for i in range(0,len(tr))]
test_target = [targets_dummies[i,:] for i in range(len(tr),len(sequences))]

print(len(train) ,"elements in train set")
print(len(test),"elements in test set")

train = np.array(train)
test = np.array(test)

train_target = np.array(train_target)
test_target = np.array(test_target)

file_to_save = open('BayesOptPred_TrainTestSets',"wb")
data_to_save = [[train, train_target], [test, test_target]]  
pickle.dump(data_to_save,file_to_save)
file_to_save.close()

X = train         
y = train_target  

# Build the ANN

def generate_model(dropout, neuronPct, neuronShrink):

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

        # Shrink neuron count for each layer
        neuronCount = int(neuronCount * neuronShrink)
        
    model.add(LSTM(neuronCount,input_shape=(look_back, n_features), dropout=dropout+0.1)) 
    model.add(Dense(5,activation='softmax')) 
    return model

# Cross validation

num_folds = 5

epochs_needed = []
acc_per_fold = []
loss_per_fold = []


# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# Define early stopping condition
earlStop = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, restore_best_weights=True, verbose = 0)

# K-fold Cross Validation model evaluation
fold_no = 1
for train_n, val_n in kfold.split(X, y):

    model = generate_model(dropout=0.1145, neuronPct=0.1012, neuronShrink=0.1793) # model H

    # Compile the model
    adam = Adam(lr=0.0018) 

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
    
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history = model.fit(X[train_n], y[train_n], validation_data=[X[val_n], y[val_n]], batch_size=32, epochs=100, verbose=0, callbacks=[earlStop])
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