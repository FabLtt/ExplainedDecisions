# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 09:44:37 2020

@author: fa17936

Scope: perform Bayesian Optimization on ANN composed by alternating LSTM and Dropout layers 
"""
# Supress NaN warnings
import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pickle   

import numpy as np
from pandas import get_dummies

import tensorflow.keras.initializers
import statistics
from sklearn import metrics

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit 
from tensorflow.keras.optimizers import Adam

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events



print("\n\n starting..")

# Open the dataset file
file = open("./dataset/DatasetFileMultiClassPred_BothHerders_WrtGoal_Extended","rb")

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
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :-1]
        dataX.append(a)
        dataY.append(dataset[i + look_back, -1])
    return dataX, dataY 


look_back = 25  # Sequence length
sequences = []
targets = []

herders_tot = int(max(Dataset[:,0]))
trial_tot = int(max(Dataset[:,1]))

# A sequence can refer to only one Herder ID and one Trial ID
for herder_id in range(herders_tot):
    for trial_id in range(trial_tot):
        Dtst = Dataset_df[(Dataset_df["Herder_id"]==herder_id) & (Dataset_df["Trial_id"]==trial_id)].values[:,2:]
        seq, tar = create_dataset(Dtst, look_back)
        sequences = sequences + seq
        targets = targets + tar

print("samples before splitting:", len(targets))
# for el in range(5):
#     print("samples in class ",el," : " , round(targets.count(el)/len(targets),2))
    
train = np.array(sequences)
train_targets = np.array(targets)
    
train_samples = 2000 # training set size

# Randomly select 'train_samples' samples from the total available ones (i.e. 'train_samples')
sss = StratifiedShuffleSplit(train_size=train_samples, n_splits=1, 
                             test_size=5, random_state=0)  

for train_index, test_index in sss.split(train, train_targets):
    X_train, X_test = train[train_index], train[test_index]
    y_train, y_test = train_targets[train_index], train_targets[test_index]
    
    
print("samples after splitting:", y_train.shape)

# Rename training samples to be fed the ANN
dummies = get_dummies(y_train)
targets_dummies = dummies.values 

x = X_train
y = targets_dummies



# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def generate_model(dropout, neuronPct, neuronShrink):
    # We start with some percent of 5000 starting neurons on 
    # the first hidden layer.
    neuronCount = int(neuronPct * 2500)
    
    # Construct neural network
    model = Sequential()

    # So long as there would have been at least 25 neurons and 
    # fewer than 10 layers, create a new layer.
    layer = 0
    while neuronCount>25 and layer<10:
        # The first (0th) layer needs an input input_dim(neuronCount)
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
    model.add(Dense(5,activation='softmax')) # Output multi
    return model


def evaluate_network(dropout,lr,neuronPct,neuronShrink):
    SPLITS = 2

    # Bootstrap
    boot = StratifiedShuffleSplit(n_splits=SPLITS, test_size=0.2)

    # Track progress
    mean_benchmark = []
    epochs_needed = []
    num = 0
    

    # Loop through samples
    for train_n, val_n in boot.split(x,y):
        # start_time = time.time()
        num+=1

        # Split train and test
        x_train = x[train_n]
        y_train = y[train_n]
        x_val = x[val_n]
        y_val = y[val_n]

        model = generate_model(dropout, neuronPct, neuronShrink)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr)) 
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
        patience=10, verbose=0, restore_best_weights=True)
        
        # Train on the bootstrap sample
        model.fit(x_train,y_train,validation_data=(x_val,y_val), callbacks=[monitor],verbose=0,epochs=100)
        epochs = monitor.stopped_epoch
        epochs_needed.append(epochs)

        # Predict on the out of boot (validation)
        pred = model.predict(x_val)

        # Measure this bootstrap's log loss
        y_compare = np.argmax(y_val,axis=1) # For log loss calculation
        score = metrics.log_loss(y_compare, pred)
        mean_benchmark.append(score)
        m1 = statistics.mean(mean_benchmark)
        
    tensorflow.keras.backend.clear_session()
    return (-m1)


# Bounded region of parameter space
pbounds = {'dropout': (0.1, 0.399),
            'lr': (0.001, 0.01),
            'neuronPct': (0.01, 0.499),
            'neuronShrink': (0.01, 0.499)
          }

optimizer = BayesianOptimization(f=evaluate_network, pbounds=pbounds, verbose=0, random_state=1)


logger = JSONLogger(path='./checkpoint/BayesOpt_logs.json')
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.maximize(init_points=10, n_iter=20,)

print(f"Best combination of parameters and target value: {optimizer.max}")
