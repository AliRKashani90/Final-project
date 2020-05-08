# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:26:57 2020

@author: alireza
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from keras.layers import Dense, Activation, Flatten
from keras import metrics
#from matplotlib import pyplot as plt
#import seaborn as sns
#import os
np.random.seed(123456)
#=======================================================================================
#                             parameters
#=======================================================================================
dir_out = r"C:\Users\alireza\Desktop\Project"
splitRatio = 1./4 # for training and test data
# FFNN hyper parameters
BPNN_hyper_main1 = {
        'unit1'       :  9, # number of neurons in a layer
        'act_f1'      :  'relu', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'unit2'       :  240, # number of neurons in a layer
        'act_f2'      :  'relu', # activation function
        'unit3'       :  1, # number of neurons in a layer
        'act_f3'      :  'relu', # activation function
        'unit4'       :  1, # number of neurons in a layer
        'act_f4'      :  'relu', # activation function
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        'epochs'      :  100, # Number of epochs
        'verbose'     :  0, # No output
        'batch_size'  :  100 # Number of observations per batch
        }
BPNN_hyper_main2 = {
        'unit1'       :  9, # number of neurons in a layer
        'act_f1'      :  'relu', # activation function
        'unit2'       :  400, # number of neurons in a layer
        'act_f2'      :  'relu', # activation function
        'unit3'       :  1, # number of neurons in a layer
        'act_f3'      :  'relu', # activation function
        'unit4'       :  80, # number of neurons in a layer
        'act_f4'      :  'relu', # activation function
        'loss'        :  'mse', # Mean squared error
        'optimizer'   :  'RMSprop', # Optimization algorithm
        'metrics'     :  ['mse'],
        'epochs'      :  100, # Number of epochs
        'verbose'     :  0, # No output
        'batch_size'  :  100 # Number of observations per batch
        }
#=======================================================================================
#                       download and prep data
#=======================================================================================
#------------------------------------------------------------------------------
############################  considering Rjb  ################################
#------------------------------------------------------------------------------
# Split dataset to train and test sets.
dData1=pd.read_csv('Data1.csv', skiprows = []).as_matrix()
X1 = dData1[:,:-1]
Y1 = dData1[:,-1]
X_train1, X_test1, Y_train1, Y_test1 = train_test_split( dData1[:,:-1], dData1[:,-1],
                                    shuffle = True, test_size = splitRatio)
print('Train: Rows: %d, columns: %d' % (X_train1.shape[0], X_train1.shape[1]))
print('Test: Rows: %d, columns: %d'%(X_test1.shape[0], X_test1.shape[1]))
# ------------- Calculate Corr. matrix
title1 = ['T','Mw','Rjb','FRV','FNM','ZTOR','Delta','Vs_30','Z']
X_dataframe1 = pd.DataFrame( X1, columns=title1)
corrMatrix1 = X_dataframe1.corr()
title2 = ['T','Mw','Rjb','FRV','FNM','ZTOR','Delta','Vs_30','Z','Ln_Sa']
dData_dataframe1 = pd.DataFrame( dData1, columns=title2)
#------------------------------------------------------------------------------
##########################  considering Rrup  #################################
#------------------------------------------------------------------------------
# Split dataset to train and test sets.
dData2=pd.read_csv('Data2.csv', skiprows = []).as_matrix()
X2 = dData2[:,:-1]
Y2 = dData2[:,-1]
X_train2, X_test2, Y_train2, Y_test2 = train_test_split( dData2[:,:-1], dData2[:,-1],
                                    shuffle = True, test_size = splitRatio)
print('Train: Rows: %d, columns: %d' % (X_train2.shape[0], X_train2.shape[1]))
print('Test: Rows: %d, columns: %d'%(X_test2.shape[0], X_test2.shape[1]))
# ------------- Calculate Corr. matrix
title1 = ['T','Mw','Rrup','FRV','FNM','ZTOR','Delta','Vs_30','Z']
X_dataframe2 = pd.DataFrame( X2, columns=title1)
corrMatrix2 = X_dataframe2.corr()
title2 = ['T','Mw','Rrup','FRV','FNM','ZTOR','Delta','Vs_30','Z','Ln_Sa']
dData_dataframe2 = pd.DataFrame( dData2, columns=title2)
#=======================================================================================
#                         BPNN Development
#=======================================================================================
def One_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper1, BPNN_hyper2):
    # ------------------- Feedforward Neural Networks with one hidden layer
    # Start neural network
    BPNN_model1 = models.Sequential()
    # Add fully connected layer with a ReLU activation function
    BPNN_model1.add(layers.Dense(units=BPNN_hyper1['unit1'], activation=BPNN_hyper2['act_f'], input_shape=(X_train1.shape[1],)))
    # Add fully connected layer with a ReLU activation function
    BPNN_model1.add(layers.Dense(units=BPNN_hyper1['unit2'], activation=BPNN_hyper2['act_f']))
    # Add fully connected layer with no activation function
    BPNN_model1.add(layers.Dense(units=BPNN_hyper1['unit3'], activation=BPNN_hyper2['act_f']))
    # Compile neural network
    BPNN_model1.compile(loss=BPNN_hyper2['loss'], # Mean squared error
                        optimizer=BPNN_hyper2['optimizer'], # Optimization algorithm
                        metrics=BPNN_hyper2['metrics']) # Mean squared error
    # Train neural network
    BPNN_model1.fit(X_train1, Y_train1, # Target vector
                    epochs=BPNN_hyper1['epochs'], # Number of epochs
                    verbose=BPNN_hyper1['verbose'], # No output
                    batch_size=BPNN_hyper1['batch_size'], # Number of observations per batch
                    validation_data=(X_test1, Y_test1)) # Data for evaluation
    return BPNN_model1
# ------------------- Feedforward Neural Networks with one hidden layer
def Two_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper1, BPNN_hyper2):
    # ------------------- Feedforward Neural Networks with one hidden layer
    # Start neural network
    BPNN_model1 = models.Sequential()
    # Add fully connected layer with a ReLU activation function
    BPNN_model1.add(layers.Dense(units=BPNN_hyper1['unit1'], activation=BPNN_hyper2['act_f'], input_shape=(X_train1.shape[1],)))
    # Add fully connected layer with a ReLU activation function
    BPNN_model1.add(layers.Dense(units=BPNN_hyper1['unit2'], activation=BPNN_hyper2['act_f']))
    # Add fully connected layer with a ReLU activation function
    BPNN_model1.add(layers.Dense(units=BPNN_hyper1['unit4'], activation=BPNN_hyper2['act_f']))
    # Add fully connected layer with no activation function
    BPNN_model1.add(layers.Dense(units=BPNN_hyper1['unit3'], activation=BPNN_hyper2['act_f']))
    # Compile neural network
    BPNN_model1.compile(loss=BPNN_hyper2['loss'], # Mean squared error
                        optimizer=BPNN_hyper2['optimizer'], # Optimization algorithm
                        metrics=BPNN_hyper2['metrics']) # Mean squared error
    # Train neural network
    BPNN_model1.fit(X_train1, Y_train1, # Target vector
                    epochs=BPNN_hyper1['epochs'], # Number of epochs
                    verbose=BPNN_hyper1['verbose'], # No output
                    batch_size=BPNN_hyper1['batch_size'], # Number of observations per batch
                    validation_data=(X_test1, Y_test1)) # Data for evaluation
    return BPNN_model1
#=======================================================================================
#                        prediction and accuracy
#=======================================================================================
#                      One-layer BPNN neural network
#------------------------------------------------------------------------------
############################  considering Rjb  ################################
#------------------------------------------------------------------------------
BPNN_hyper1 = {
        'act_f'      :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model1 = One_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper1)
y_pred_FFNN_one_1Rjb = FFNN_model1.predict(X_test1) # Feedforward Neural Networks
acc_FFNN_one_1Rjb = metrics.accuracy(Y_test1, y_pred_FFNN_one_1Rjb)

BPNN_hyper2 = {
        'act_f'       :  'sigmoid', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model2 = One_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper2)
y_pred_FFNN_one_2Rjb = FFNN_model2.predict(X_test1) # Feedforward Neural Networks
acc_FFNN_one_2Rjb = metrics.accuracy(Y_test1, y_pred_FFNN_one_2Rjb)

BPNN_hyper3 = {
        'act_f'       :  'softmax', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model3 = One_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper3)
y_pred_FFNN_one_3Rjb = FFNN_model3.predict(X_test1) # Feedforward Neural Networks
acc_FFNN_one_3Rjb = metrics.accuracy(Y_test1, y_pred_FFNN_one_3Rjb)

BPNN_hyper4 = {
        'act_f'       :  'tanh', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model4 = One_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper4)
y_pred_FFNN_one_4Rjb = FFNN_model4.predict(X_test1) # Feedforward Neural Networks
acc_FFNN_one_4Rjb = metrics.accuracy(Y_test1, y_pred_FFNN_one_4Rjb)

BPNN_hyper5 = {
        'act_f'       :  'relu', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model5 = One_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper5)
y_pred_FFNN_one_5Rjb = FFNN_model5.predict(X_test1) # Feedforward Neural Networks
acc_FFNN_one_5Rjb = metrics.accuracy(Y_test1, y_pred_FFNN_one_5Rjb)

BPNN_hyper6 = {
        'act_f'       :  'elu', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model6 = One_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper6)
y_pred_FFNN_one_6Rjb = FFNN_model6.predict(X_test1) # Feedforward Neural Networks
acc_FFNN_one_6Rjb = metrics.accuracy(Y_test1, y_pred_FFNN_one_6Rjb)

BPNN_hyper7 = {
        'act_f'       :  'selu', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model7 = One_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper7)
y_pred_FFNN_one_7Rjb = FFNN_model7.predict(X_test1) # Feedforward Neural Networks
acc_FFNN_one_7Rjb = metrics.accuracy(Y_test1, y_pred_FFNN_one_7Rjb)

BPNN_hyper8 = {
        'act_f'       :  'softplus', # activation function  'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model8 = One_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper8)
y_pred_FFNN_one_8Rjb = FFNN_model8.predict(X_test1) # Feedforward Neural Networks
acc_FFNN_one_8Rjb = metrics.accuracy(Y_test1, y_pred_FFNN_one_8Rjb)

BPNN_hyper9 = {
        'act_f'       :  'softsign', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model9 = One_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper9)
y_pred_FFNN_one_9Rjb = FFNN_model9.predict(X_test1) # Feedforward Neural Networks
acc_FFNN_one_9Rjb = metrics.accuracy(Y_test1, y_pred_FFNN_one_9Rjb)

BPNN_hyper10 = {
        'act_f'       :  'hard_sigmoid', # activation function 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model10 = One_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper10)
y_pred_FFNN_one_10Rjb = FFNN_model10.predict(X_test1) # Feedforward Neural Networks
acc_FFNN_one_10Rjb = metrics.accuracy(Y_test1, y_pred_FFNN_one_10Rjb)

BPNN_hyper11 = {
        'act_f'       :  'exponential', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model11 = One_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper11)
y_pred_FFNN_one_11Rjb = FFNN_model11.predict(X_test1) # Feedforward Neural Networks
acc_FFNN_one_11Rjb = metrics.accuracy(Y_test1, y_pred_FFNN_one_11Rjb)
#------------------------------------------------------------------------------
############################  considering Rrup  ###############################
#------------------------------------------------------------------------------
# One-layer BPNN neural network
BPNN_hyper1 = {
        'act_f'      :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model1 = One_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper1)
y_pred_FFNN_one_1Rrup = FFNN_model1.predict(X_test2) # Feedforward Neural Networks
acc_FFNN_one_1Rrup = metrics.accuracy(Y_test2, y_pred_FFNN_one_1Rrup)

BPNN_hyper2 = {
        'act_f'       :  'sigmoid', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model2 = One_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper2)
y_pred_FFNN_one_2Rrup = FFNN_model2.predict(X_test2) # Feedforward Neural Networks
acc_FFNN_one_2Rrup = metrics.accuracy(Y_test2, y_pred_FFNN_one_2Rrup)

BPNN_hyper3 = {
        'act_f'       :  'softmax', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model3 = One_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper3)
y_pred_FFNN_one_3Rrup = FFNN_model3.predict(X_test2) # Feedforward Neural Networks
acc_FFNN_one_3Rrup = metrics.accuracy(Y_test2, y_pred_FFNN_one_3Rrup)

BPNN_hyper4 = {
        'act_f'       :  'tanh', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model4 = One_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper4)
y_pred_FFNN_one_4Rrup = FFNN_model4.predict(X_test2) # Feedforward Neural Networks
acc_FFNN_one_4Rrup = metrics.accuracy(Y_test2, y_pred_FFNN_one_4Rrup)

BPNN_hyper5 = {
        'act_f'       :  'relu', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model5 = One_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper5)
y_pred_FFNN_one_5Rrup = FFNN_model5.predict(X_test2) # Feedforward Neural Networks
acc_FFNN_one_5Rrup = metrics.accuracy(Y_test2, y_pred_FFNN_one_5Rrup)

BPNN_hyper6 = {
        'act_f'       :  'elu', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model6 = One_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper6)
y_pred_FFNN_one_6Rrup = FFNN_model6.predict(X_test2) # Feedforward Neural Networks
acc_FFNN_one_6Rrup = metrics.accuracy(Y_test2, y_pred_FFNN_one_6Rrup)

BPNN_hyper7 = {
        'act_f'       :  'selu', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model7 = One_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper7)
y_pred_FFNN_one_7Rrup = FFNN_model7.predict(X_test2) # Feedforward Neural Networks
acc_FFNN_one_7Rrup = metrics.accuracy(Y_test2, y_pred_FFNN_one_7Rrup)

BPNN_hyper8 = {
        'act_f'       :  'softplus', # activation function  'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model8 = One_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper8)
y_pred_FFNN_one_8Rrup = FFNN_model8.predict(X_test2) # Feedforward Neural Networks
acc_FFNN_one_8Rrup = metrics.accuracy(Y_test2, y_pred_FFNN_one_8Rrup)

BPNN_hyper9 = {
        'act_f'       :  'softsign', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model9 = One_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper9)
y_pred_FFNN_one_9Rrup = FFNN_model9.predict(X_test2) # Feedforward Neural Networks
acc_FFNN_one_9Rrup = metrics.accuracy(Y_test2, y_pred_FFNN_one_9Rrup)

BPNN_hyper10 = {
        'act_f'       :  'hard_sigmoid', # activation function 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model10 = One_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper10)
y_pred_FFNN_one_10Rrup = FFNN_model10.predict(X_test2) # Feedforward Neural Networks
acc_FFNN_one_10Rrup = metrics.accuracy(Y_test2, y_pred_FFNN_one_10Rrup)

BPNN_hyper11 = {
        'act_f'       :  'exponential', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model11 = One_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper11)
y_pred_FFNN_one_11Rrup = FFNN_model11.predict(X_test2) # Feedforward Neural Networks
acc_FFNN_one_11Rrup = metrics.accuracy(Y_test2, y_pred_FFNN_one_11Rrup)
###############################################################################
###############################################################################
###############################################################################
###############################################################################
#                      Two-layer BPNN neural network
#------------------------------------------------------------------------------
############################  considering Rjb  ################################
#------------------------------------------------------------------------------
BPNN_hyper1 = {
        'act_f'      :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model1 = Two_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper1)
y_pred_FFNN_Two_1Rjb = FFNN_model1.predict(X_test1) # Feedforward Neural Networks
acc_FFNN_Two_1Rjb = metrics.accuracy(Y_test1, y_pred_FFNN_Two_1Rjb)

BPNN_hyper2 = {
        'act_f'       :  'sigmoid', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model2 = Two_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper2)
y_pred_FFNN_Two_2Rjb = FFNN_model2.predict(X_test1) # Feedforward Neural Networks
acc_FFNN_Two_2Rjb = metrics.accuracy(Y_test1, y_pred_FFNN_Two_2Rjb)

BPNN_hyper3 = {
        'act_f'       :  'softmax', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model3 = Two_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper3)
y_pred_FFNN_Two_3Rjb = FFNN_model3.predict(X_test1) # Feedforward Neural Networks
acc_FFNN_Two_3Rjb = metrics.accuracy(Y_test1, y_pred_FFNN_Two_3Rjb)

BPNN_hyper4 = {
        'act_f'       :  'tanh', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model4 = Two_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper4)
y_pred_FFNN_Two_4Rjb = FFNN_model4.predict(X_test1) # Feedforward Neural Networks
acc_FFNN_Two_4Rjb = metrics.accuracy(Y_test1, y_pred_FFNN_Two_4Rjb)

BPNN_hyper5 = {
        'act_f'       :  'relu', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model5 = Two_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper5)
y_pred_FFNN_Two_5Rjb = FFNN_model5.predict(X_test1) # Feedforward Neural Networks
acc_FFNN_Two_5Rjb = metrics.accuracy(Y_test1, y_pred_FFNN_Two_5Rjb)

BPNN_hyper6 = {
        'act_f'       :  'elu', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model6 = Two_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper6)
y_pred_FFNN_Two_6Rjb = FFNN_model6.predict(X_test1) # Feedforward Neural Networks
acc_FFNN_Two_6Rjb = metrics.accuracy(Y_test1, y_pred_FFNN_Two_6Rjb)

BPNN_hyper7 = {
        'act_f'       :  'selu', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model7 = Two_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper7)
y_pred_FFNN_Two_7Rjb = FFNN_model7.predict(X_test1) # Feedforward Neural Networks
acc_FFNN_Two_7Rjb = metrics.accuracy(Y_test1, y_pred_FFNN_Two_7Rjb)

BPNN_hyper8 = {
        'act_f'       :  'softplus', # activation function  'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model8 = Two_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper8)
y_pred_FFNN_Two_8Rjb = FFNN_model8.predict(X_test1) # Feedforward Neural Networks
acc_FFNN_Two_8Rjb = metrics.accuracy(Y_test1, y_pred_FFNN_Two_8Rjb)

BPNN_hyper9 = {
        'act_f'       :  'softsign', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model9 = Two_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper9)
y_pred_FFNN_Two_9Rjb = FFNN_model9.predict(X_test1) # Feedforward Neural Networks
acc_FFNN_Two_9Rjb = metrics.accuracy(Y_test1, y_pred_FFNN_Two_9Rjb)

BPNN_hyper10 = {
        'act_f'       :  'hard_sigmoid', # activation function 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model10 = Two_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper10)
y_pred_FFNN_Two_10Rjb = FFNN_model10.predict(X_test1) # Feedforward Neural Networks
acc_FFNN_Two_10Rjb = metrics.accuracy(Y_test1, y_pred_FFNN_Two_10Rjb)

BPNN_hyper11 = {
        'act_f'       :  'exponential', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model11 = Two_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper11)
y_pred_FFNN_Two_11Rjb = FFNN_model11.predict(X_test1) # Feedforward Neural Networks
acc_FFNN_Two_11Rjb = metrics.accuracy(Y_test1, y_pred_FFNN_Two_11Rjb)
#------------------------------------------------------------------------------
############################  considering Rrup  ###############################
#------------------------------------------------------------------------------
# One-layer BPNN neural network
BPNN_hyper1 = {
        'act_f'      :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model1 = Two_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper1)
y_pred_FFNN_Two_1Rrup = FFNN_model1.predict(X_test2) # Feedforward Neural Networks
acc_FFNN_Two_1Rrup = metrics.accuracy(Y_test2, y_pred_FFNN_Two_1Rrup)

BPNN_hyper2 = {
        'act_f'       :  'sigmoid', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model2 = Two_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper2)
y_pred_FFNN_Two_2Rrup = FFNN_model2.predict(X_test2) # Feedforward Neural Networks
acc_FFNN_Two_2Rrup = metrics.accuracy(Y_test2, y_pred_FFNN_Two_2Rrup)

BPNN_hyper3 = {
        'act_f'       :  'softmax', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model3 = Two_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper3)
y_pred_FFNN_Two_3Rrup = FFNN_model3.predict(X_test2) # Feedforward Neural Networks
acc_FFNN_Two_3Rrup = metrics.accuracy(Y_test2, y_pred_FFNN_Two_3Rrup)

BPNN_hyper4 = {
        'act_f'       :  'tanh', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model4 = Two_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper4)
y_pred_FFNN_Two_4Rrup = FFNN_model4.predict(X_test2) # Feedforward Neural Networks
acc_FFNN_Two_4Rrup = metrics.accuracy(Y_test2, y_pred_FFNN_Two_4Rrup)

BPNN_hyper5 = {
        'act_f'       :  'relu', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model5 = Two_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper5)
y_pred_FFNN_Two_5Rrup = FFNN_model5.predict(X_test2) # Feedforward Neural Networks
acc_FFNN_Two_5Rrup = metrics.accuracy(Y_test2, y_pred_FFNN_Two_5Rrup)

BPNN_hyper6 = {
        'act_f'       :  'elu', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model6 = Two_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper6)
y_pred_FFNN_Two_6Rrup = FFNN_model6.predict(X_test2) # Feedforward Neural Networks
acc_FFNN_Two_6Rrup = metrics.accuracy(Y_test2, y_pred_FFNN_Two_6Rrup)

BPNN_hyper7 = {
        'act_f'       :  'selu', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model7 = Two_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper7)
y_pred_FFNN_Two_7Rrup = FFNN_model7.predict(X_test2) # Feedforward Neural Networks
acc_FFNN_Two_7Rrup = metrics.accuracy(Y_test2, y_pred_FFNN_Two_7Rrup)

BPNN_hyper8 = {
        'act_f'       :  'softplus', # activation function  'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model8 = Two_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper8)
y_pred_FFNN_Two_8Rrup = FFNN_model8.predict(X_test2) # Feedforward Neural Networks
acc_FFNN_Two_8Rrup = metrics.accuracy(Y_test2, y_pred_FFNN_Two_8Rrup)

BPNN_hyper9 = {
        'act_f'       :  'softsign', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model9 = Two_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper9)
y_pred_FFNN_Two_9Rrup = FFNN_model9.predict(X_test2) # Feedforward Neural Networks
acc_FFNN_Two_9Rrup = metrics.accuracy(Y_test2, y_pred_FFNN_Two_9Rrup)

BPNN_hyper10 = {
        'act_f'       :  'hard_sigmoid', # activation function 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model10 = Two_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper10)
y_pred_FFNN_Two_10Rrup = FFNN_model10.predict(X_test2) # Feedforward Neural Networks
acc_FFNN_Two_10Rrup = metrics.accuracy(Y_test2, y_pred_FFNN_Two_10Rrup)

BPNN_hyper11 = {
        'act_f'       :  'exponential', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model11 = Two_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper11)
y_pred_FFNN_Two_11Rrup = FFNN_model11.predict(X_test2) # Feedforward Neural Networks
acc_FFNN_Two_11Rrup = metrics.accuracy(Y_test2, y_pred_FFNN_Two_11Rrup)
###############################################################################
###############################################################################
###############################################################################

#                      One-layer BPNN neural network
#------------------------------------------------------------------------------
############################  considering Rjb  ################################
#------------------------------------------------------------------------------
BPNN_hyper1 = {
        'act_f'      :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'SGD', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model1 = One_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper1)
OPT_y_pred_FFNN_one_1Rjb = FFNN_model1.predict(X_test1) # Feedforward Neural Networks
OPT_acc_FFNN_one_1Rjb = metrics.accuracy(Y_test1, OPT_y_pred_FFNN_one_1Rjb)

BPNN_hyper2 = {
        'act_f'       :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'Adagrad', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model2 = One_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper2)
OPT_y_pred_FFNN_one_2Rjb = FFNN_model2.predict(X_test1) # Feedforward Neural Networks
OPT_acc_FFNN_one_2Rjb = metrics.accuracy(Y_test1, OPT_y_pred_FFNN_one_2Rjb)

BPNN_hyper3 = {
        'act_f'       :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'Adadelta', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model3 = One_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper3)
OPT_y_pred_FFNN_one_3Rjb = FFNN_model3.predict(X_test1) # Feedforward Neural Networks
OPT_acc_FFNN_one_3Rjb = metrics.accuracy(Y_test1, OPT_y_pred_FFNN_one_3Rjb)

BPNN_hyper4 = {
        'act_f'       :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'Adam', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model4 = One_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper4)
OPT_y_pred_FFNN_one_4Rjb = FFNN_model4.predict(X_test1) # Feedforward Neural Networks
OPT_acc_FFNN_one_4Rjb = metrics.accuracy(Y_test1, OPT_y_pred_FFNN_one_4Rjb)

BPNN_hyper5 = {
        'act_f'       :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'Adamax', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model5 = One_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper5)
OPT_y_pred_FFNN_one_5Rjb = FFNN_model5.predict(X_test1) # Feedforward Neural Networks
OPT_acc_FFNN_one_5Rjb = metrics.accuracy(Y_test1, OPT_y_pred_FFNN_one_5Rjb)

BPNN_hyper6 = {
        'act_f'       :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'Nadam', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model6 = One_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper6)
OPT_y_pred_FFNN_one_6Rjb = FFNN_model6.predict(X_test1) # Feedforward Neural Networks
OPT_acc_FFNN_one_6Rjb = metrics.accuracy(Y_test1, OPT_y_pred_FFNN_one_6Rjb)

BPNN_hyper7 = {
        'act_f'       :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model7 = One_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper7)
OPT_y_pred_FFNN_one_7Rjb = FFNN_model7.predict(X_test1) # Feedforward Neural Networks
OPT_acc_FFNN_one_7Rjb = metrics.accuracy(Y_test1, OPT_y_pred_FFNN_one_7Rjb)
#------------------------------------------------------------------------------
############################  considering Rrup  ###############################
#------------------------------------------------------------------------------
# One-layer BPNN neural network
BPNN_hyper1 = {
        'act_f'      :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'SGD', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model1 = One_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper1)
OPT_y_pred_FFNN_one_1Rrup = FFNN_model1.predict(X_test2) # Feedforward Neural Networks
OPT_acc_FFNN_one_1Rrup = metrics.accuracy(Y_test2, OPT_y_pred_FFNN_one_1Rrup)

BPNN_hyper2 = {
        'act_f'       :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'Adagrad', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model2 = One_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper2)
OPT_y_pred_FFNN_one_2Rrup = FFNN_model2.predict(X_test2) # Feedforward Neural Networks
OPT_acc_FFNN_one_2Rrup = metrics.accuracy(Y_test2, OPT_y_pred_FFNN_one_2Rrup)

BPNN_hyper3 = {
        'act_f'       :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'Adadelta', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model3 = One_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper3)
OPT_y_pred_FFNN_one_3Rrup = FFNN_model3.predict(X_test2) # Feedforward Neural Networks
OPT_acc_FFNN_one_3Rrup = metrics.accuracy(Y_test2, OPT_y_pred_FFNN_one_3Rrup)

BPNN_hyper4 = {
        'act_f'       :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'Adam', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model4 = One_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper4)
OPT_y_pred_FFNN_one_4Rrup = FFNN_model4.predict(X_test2) # Feedforward Neural Networks
OPT_acc_FFNN_one_4Rrup = metrics.accuracy(Y_test2, OPT_y_pred_FFNN_one_4Rrup)

BPNN_hyper5 = {
        'act_f'       :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'Adamax', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model5 = One_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper5)
OPT_y_pred_FFNN_one_5Rrup = FFNN_model5.predict(X_test2) # Feedforward Neural Networks
OPT_acc_FFNN_one_5Rrup = metrics.accuracy(Y_test2, OPT_y_pred_FFNN_one_5Rrup)

BPNN_hyper6 = {
        'act_f'       :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'Nadam', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model6 = One_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper6)
OPT_y_pred_FFNN_one_6Rrup = FFNN_model6.predict(X_test2) # Feedforward Neural Networks
OPT_acc_FFNN_one_6Rrup = metrics.accuracy(Y_test2, OPT_y_pred_FFNN_one_6Rrup)

BPNN_hyper7 = {
        'act_f'       :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model7 = One_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper7)
OPT_y_pred_FFNN_one_7Rrup = FFNN_model7.predict(X_test2) # Feedforward Neural Networks
OPT_acc_FFNN_one_7Rrup = metrics.accuracy(Y_test2, OPT_y_pred_FFNN_one_7Rrup)
###############################################################################
###############################################################################
###############################################################################
#                      Two-layer BPNN neural network
#------------------------------------------------------------------------------
############################  considering Rjb  ################################
#------------------------------------------------------------------------------
BPNN_hyper1 = {
        'act_f'       :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'SGD', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model1 = Two_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper1)
OPT_y_pred_FFNN_Two_1Rjb = FFNN_model1.predict(X_test1) # Feedforward Neural Networks
OPT_acc_FFNN_Two_1Rjb = metrics.accuracy(Y_test1, OPT_y_pred_FFNN_Two_1Rjb)

BPNN_hyper2 = {
        'act_f'       :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'Adagrad', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model2 = Two_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper2)
OPT_y_pred_FFNN_Two_2Rjb = FFNN_model2.predict(X_test1) # Feedforward Neural Networks
OPT_acc_FFNN_Two_2Rjb = metrics.accuracy(Y_test1, OPT_y_pred_FFNN_Two_2Rjb)

BPNN_hyper3 = {
        'act_f'       :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'Adadelta', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model3 = Two_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper3)
OPT_y_pred_FFNN_Two_3Rjb = FFNN_model3.predict(X_test1) # Feedforward Neural Networks
OPT_acc_FFNN_Two_3Rjb = metrics.accuracy(Y_test1, OPT_y_pred_FFNN_Two_3Rjb)

BPNN_hyper4 = {
        'act_f'       :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'Adam', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model4 = Two_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper4)
OPT_y_pred_FFNN_Two_4Rjb = FFNN_model4.predict(X_test1) # Feedforward Neural Networks
OPT_acc_FFNN_Two_4Rjb = metrics.accuracy(Y_test1, OPT_y_pred_FFNN_Two_4Rjb)

BPNN_hyper5 = {
        'act_f'       :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'Adamax', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model5 = Two_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper5)
OPT_y_pred_FFNN_Two_5Rjb = FFNN_model5.predict(X_test1) # Feedforward Neural Networks
OPT_acc_FFNN_Two_5Rjb = metrics.accuracy(Y_test1, OPT_y_pred_FFNN_Two_5Rjb)

BPNN_hyper6 = {
        'act_f'       :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'Nadam', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model6 = Two_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper6)
OPT_y_pred_FFNN_Two_6Rjb = FFNN_model6.predict(X_test1) # Feedforward Neural Networks
OPT_acc_FFNN_Two_6Rjb = metrics.accuracy(Y_test1, OPT_y_pred_FFNN_Two_6Rjb)

BPNN_hyper7 = {
        'act_f'       :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model7 = Two_layer_BPNN_RJB ( X_train1, Y_train1, X_test1, Y_test1, BPNN_hyper_main1, BPNN_hyper7)
OPT_y_pred_FFNN_Two_7Rjb = FFNN_model7.predict(X_test1) # Feedforward Neural Networks
OPT_acc_FFNN_Two_7Rjb = metrics.accuracy(Y_test1, OPT_y_pred_FFNN_Two_7Rjb)
#------------------------------------------------------------------------------
############################  considering Rrup  ###############################
#------------------------------------------------------------------------------
# One-layer BPNN neural network
BPNN_hyper1 = {
        'act_f'      :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'SGD', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model1 = Two_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper1)
OPT_y_pred_FFNN_Two_1Rrup = FFNN_model1.predict(X_test2) # Feedforward Neural Networks
OPT_acc_FFNN_Two_1Rrup = metrics.accuracy(Y_test2, OPT_y_pred_FFNN_Two_1Rrup)

BPNN_hyper2 = {
        'act_f'       :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'Adagrad', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model2 = Two_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper2)
OPT_y_pred_FFNN_Two_2Rrup = FFNN_model2.predict(X_test2) # Feedforward Neural Networks
OPT_acc_FFNN_Two_2Rrup = metrics.accuracy(Y_test2, OPT_y_pred_FFNN_Two_2Rrup)

BPNN_hyper3 = {
        'act_f'       :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'Adadelta', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model3 = Two_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper3)
OPT_y_pred_FFNN_Two_3Rrup = FFNN_model3.predict(X_test2) # Feedforward Neural Networks
OPT_acc_FFNN_Two_3Rrup = metrics.accuracy(Y_test2, OPT_y_pred_FFNN_Two_3Rrup)

BPNN_hyper4 = {
        'act_f'       :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'Adam', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model4 = Two_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper4)
OPT_y_pred_FFNN_Two_4Rrup = FFNN_model4.predict(X_test2) # Feedforward Neural Networks
OPT_acc_FFNN_Two_4Rrup = metrics.accuracy(Y_test2, OPT_y_pred_FFNN_Two_4Rrup)

BPNN_hyper5 = {
        'act_f'       :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'Adamax', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model5 = Two_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper5)
OPT_y_pred_FFNN_Two_5Rrup = FFNN_model5.predict(X_test2) # Feedforward Neural Networks
OPT_acc_FFNN_Two_5Rrup = metrics.accuracy(Y_test2, OPT_y_pred_FFNN_Two_5Rrup)

BPNN_hyper6 = {
        'act_f'       :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'Nadam', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model6 = Two_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper6)
OPT_y_pred_FFNN_Two_6Rrup = FFNN_model6.predict(X_test2) # Feedforward Neural Networks
OPT_acc_FFNN_Two_6Rrup = metrics.accuracy(Y_test2, OPT_y_pred_FFNN_Two_6Rrup)

BPNN_hyper7 = {
        'act_f'       :  'linear', # activation function 'linear', 'sigmoid', 'softmax', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential'
        'loss'        :  'mse', # Mean squared error 
        'optimizer'   :  'RMSprop', # Optimization algorithm 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
        'metrics'     :  ['mse'], # 'mae'
        }
FFNN_model7 = Two_layer_BPNN_RJB ( X_train2, Y_train2, X_test2, Y_test2, BPNN_hyper_main1, BPNN_hyper7)
OPT_y_pred_FFNN_Two_7Rrup = FFNN_model7.predict(X_test2) # Feedforward Neural Networks
OPT_acc_FFNN_Two_7Rrup = metrics.accuracy(Y_test2, OPT_y_pred_FFNN_Two_7Rrup)