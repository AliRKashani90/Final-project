# Basic packages
import numpy as np
# Data preparation packages
import pandas as pd
from sklearn.model_selection import train_test_split
# ANN related packages
from keras import models
from keras import layers
# PSO related packages
import pyswarms as ps
from pyswarms.backend.topology import Star
import sklearn.metrics as skMetric
np.random.seed(123456)
#=======================================================================================
#                             parameters
#=======================================================================================
dir_out = r"C:\Users\alireza\Desktop\Project"
splitRatio = 1./4 # for training and test data
scaling_factor = 5 # Scaling factor for initial setting of number of neurons
# MLP hyper parameters
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
corrMatrix = X_dataframe1.corr()
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
corrMatrix = X_dataframe2.corr()
title2 = ['T','Mw','Rrup','FRV','FNM','ZTOR','Delta','Vs_30','Z','Ln_Sa']
dData_dataframe2 = pd.DataFrame( dData2, columns=title2)
#=======================================================================================
#                         Feedforward Neural Network
#=======================================================================================
# Data based on Rjb
#------------------------------------------------------------------------------
#######################  considering 1 hidden layer  ##########################
#------------------------------------------------------------------------------
def F1_1 (opt_var):
    unit2 = np.zeros((opt_var.shape[0],1))
    fAcc_test = np.zeros((1, opt_var.shape[0]))
    for i in range(opt_var.shape[0]):
        unit2[i]=opt_var[i][0]
    for i in range(opt_var.shape[0]):
        fAcc_test[0, i] = FFN_PSO1_1 ( unit2[i])
    return fAcc_test
def FFN_PSO1_1 ( par):
    if np.isnan(par[0]):
        fAcc_test =1e9
    else:
        unit2 = int(par[0])
        #    print('\n')
        #    print(unit2)
        # FFNN hyper parameters
        FFNN_hyper = {
                'unit1'       :  9, # number of neurons in a layer
                'act_f1'      :  'relu', # activation function
                'unit2'       :  32, # number of neurons in a layer
                'act_f2'      :  'relu', # activation function
                'unit3'       :  1, # number of neurons in a layer
                'loss'        :  'mse', # Mean squared error
                'optimizer'   :  'RMSprop', # Optimization algorithm
                'metrics'     :  ['mse'],
                'epochs'      :  10, # Number of epochs
                'verbose'     :  0, # No output
                'batch_size'  :  100 # Number of observations per batch
                }
        # Start neural network
        FFNN_model = models.Sequential()
        # Add fully connected layer with a ReLU activation function
        FFNN_model.add(layers.Dense(units=FFNN_hyper['unit1'], activation=FFNN_hyper['act_f1'], input_shape=(X_train1.shape[1],)))
        # Add fully connected layer with a ReLU activation function
        FFNN_model.add(layers.Dense(units= unit2, activation=FFNN_hyper['act_f2']))
        # Add fully connected layer with no activation function
        FFNN_model.add(layers.Dense(units=FFNN_hyper['unit3']))
        # Compile neural network
        FFNN_model.compile(loss=FFNN_hyper['loss'], # Mean squared error
                           optimizer=FFNN_hyper['optimizer'], # Optimization algorithm
                           metrics=FFNN_hyper['metrics']) # Mean squared error
        # Train neural network
        FFNN_model.fit(X_train1, # Features
                       Y_train1, # Target vector
                       epochs=FFNN_hyper['epochs'], # Number of epochs
                       verbose=FFNN_hyper['verbose'], # No output
                       batch_size=FFNN_hyper['batch_size'], # Number of observations per batch
                       validation_data=(X_test1, Y_test1)) # Data for evaluation
        y_pred_FFNN = FFNN_model.predict(X_test1) # Feedforward Neural Networks
        fAcc_test = skMetric.mean_squared_error( Y_test1, y_pred_FFNN)
        return fAcc_test
#------------------------------------------------------------------------------
#######################  considering 2 hidden layers  #########################
#------------------------------------------------------------------------------
def F2_1 (opt_var):
    fAcc_test = np.zeros((1, opt_var.shape[0]))
    for i in range(opt_var.shape[0]):
        fAcc_test = FFN_PSO2_1 ( opt_var[i])
    return fAcc_test
def FFN_PSO2_1 ( par):
    if np.isnan(par[0]) and np.isnan(par[1]):
        fAcc_test =1e9
    else:
        # FFNN hyper parameters
        FFNN_hyper = {
                'unit1'       :  9, # number of neurons in a layer
                'act_f1'      :  'relu', # activation function
                'unit2'       :  32, # number of neurons in a layer
                'act_f2'      :  'relu', # activation function
                'unit3'       :  1, # number of neurons in a layer
                'loss'        :  'mse', # Mean squared error
                'optimizer'   :  'RMSprop', # Optimization algorithm
                'metrics'     :  ['mse'],
                'epochs'      :  10, # Number of epochs
                'verbose'     :  0, # No output
                'batch_size'  :  100 # Number of observations per batch
                }
        # Start neural network
        FFNN_model = models.Sequential()
        # Add fully connected layer with a ReLU activation function
        FFNN_model.add(layers.Dense(units=FFNN_hyper['unit1'], activation=FFNN_hyper['act_f1'], input_shape=(X_train1.shape[1],)))
        # Add fully connected layer with a ReLU activation function
        FFNN_model.add(layers.Dense(units=int(par[0]), activation=FFNN_hyper['act_f2']))
        # Add fully connected layer with a ReLU activation function
        FFNN_model.add(layers.Dense(units=int(par[1]), activation=FFNN_hyper['act_f2']))
        # Add fully connected layer with no activation function
        FFNN_model.add(layers.Dense(units=FFNN_hyper['unit3']))
        # Compile neural network
        FFNN_model.compile(loss=FFNN_hyper['loss'], # Mean squared error
                           optimizer=FFNN_hyper['optimizer'], # Optimization algorithm
                           metrics=FFNN_hyper['metrics']) # Mean squared error
        # Train neural network
        FFNN_model.fit(X_train1, # Features
                       Y_train1, # Target vector
                       epochs=FFNN_hyper['epochs'], # Number of epochs
                       verbose=FFNN_hyper['verbose'], # No output
                       batch_size=FFNN_hyper['batch_size'], # Number of observations per batch
                       validation_data=(X_test1, Y_test1)) # Data for evaluation
        y_pred_FFNN = FFNN_model.predict(X_test1) # Feedforward Neural Networks
        fAcc_test = skMetric.mean_squared_error( Y_test1, y_pred_FFNN)
        return fAcc_test
# Data based on Rrup
#------------------------------------------------------------------------------
#######################  considering 1 hidden layer  ##########################
#------------------------------------------------------------------------------
def F1_2 (opt_var):
    unit2 = np.zeros((opt_var.shape[0],1))
    fAcc_test = np.zeros((1, opt_var.shape[0]))
    for i in range(opt_var.shape[0]):
        unit2[i]=opt_var[i][0]
    for i in range(opt_var.shape[0]):
        fAcc_test[0, i] = FFN_PSO1_2 ( unit2[i])
    return fAcc_test
def FFN_PSO1_2 ( par):
    if np.isnan(par[0]):
        fAcc_test =1e9
    else:
        unit2 = int(par[0])
        #    print('\n')
        #    print(unit2)
        # FFNN hyper parameters
        FFNN_hyper = {
                'unit1'       :  9, # number of neurons in a layer
                'act_f1'      :  'relu', # activation function
                'unit2'       :  32, # number of neurons in a layer
                'act_f2'      :  'relu', # activation function
                'unit3'       :  1, # number of neurons in a layer
                'loss'        :  'mse', # Mean squared error
                'optimizer'   :  'RMSprop', # Optimization algorithm
                'metrics'     :  ['mse'],
                'epochs'      :  10, # Number of epochs
                'verbose'     :  0, # No output
                'batch_size'  :  100 # Number of observations per batch
                }
        # Start neural network
        FFNN_model = models.Sequential()
        # Add fully connected layer with a ReLU activation function
        FFNN_model.add(layers.Dense(units=FFNN_hyper['unit1'], activation=FFNN_hyper['act_f1'], input_shape=(X_train2.shape[1],)))
        # Add fully connected layer with a ReLU activation function
        FFNN_model.add(layers.Dense(units= unit2, activation=FFNN_hyper['act_f2']))
        # Add fully connected layer with no activation function
        FFNN_model.add(layers.Dense(units=FFNN_hyper['unit3']))
        # Compile neural network
        FFNN_model.compile(loss=FFNN_hyper['loss'], # Mean squared error
                           optimizer=FFNN_hyper['optimizer'], # Optimization algorithm
                           metrics=FFNN_hyper['metrics']) # Mean squared error
        # Train neural network
        FFNN_model.fit(X_train2, # Features
                       Y_train2, # Target vector
                       epochs=FFNN_hyper['epochs'], # Number of epochs
                       verbose=FFNN_hyper['verbose'], # No output
                       batch_size=FFNN_hyper['batch_size'], # Number of observations per batch
                       validation_data=(X_test2, Y_test2)) # Data for evaluation
        y_pred_FFNN = FFNN_model.predict(X_test2) # Feedforward Neural Networks
        fAcc_test = skMetric.mean_squared_error( Y_test2, y_pred_FFNN)
        return fAcc_test
#------------------------------------------------------------------------------
#######################  considering 2 hidden layers  #########################
#------------------------------------------------------------------------------
def F2_2 (opt_var):
    fAcc_test = np.zeros((1, opt_var.shape[0]))
    for i in range(opt_var.shape[0]):
        fAcc_test = FFN_PSO2_2 ( opt_var[i])
    return fAcc_test
def FFN_PSO2_2 ( par):
    if np.isnan(par[0]) and np.isnan(par[1]):
        fAcc_test =1e9
    else:
        # FFNN hyper parameters
        FFNN_hyper = {
                'unit1'       :  9, # number of neurons in a layer
                'act_f1'      :  'relu', # activation function
                'unit2'       :  32, # number of neurons in a layer
                'act_f2'      :  'relu', # activation function
                'unit3'       :  1, # number of neurons in a layer
                'loss'        :  'mse', # Mean squared error
                'optimizer'   :  'RMSprop', # Optimization algorithm
                'metrics'     :  ['mse'],
                'epochs'      :  10, # Number of epochs
                'verbose'     :  0, # No output
                'batch_size'  :  100 # Number of observations per batch
                }
        # Start neural network
        FFNN_model = models.Sequential()
        # Add fully connected layer with a ReLU activation function
        FFNN_model.add(layers.Dense(units=FFNN_hyper['unit1'], activation=FFNN_hyper['act_f1'], input_shape=(X_train2.shape[1],)))
        # Add fully connected layer with a ReLU activation function
        FFNN_model.add(layers.Dense(units=int(par[0]), activation=FFNN_hyper['act_f2']))
        # Add fully connected layer with a ReLU activation function
        FFNN_model.add(layers.Dense(units=int(par[1]), activation=FFNN_hyper['act_f2']))
        # Add fully connected layer with no activation function
        FFNN_model.add(layers.Dense(units=FFNN_hyper['unit3']))
        # Compile neural network
        FFNN_model.compile(loss=FFNN_hyper['loss'], # Mean squared error
                           optimizer=FFNN_hyper['optimizer'], # Optimization algorithm
                           metrics=FFNN_hyper['metrics']) # Mean squared error
        # Train neural network
        FFNN_model.fit(X_train2, # Features
                       Y_train2, # Target vector
                       epochs=FFNN_hyper['epochs'], # Number of epochs
                       verbose=FFNN_hyper['verbose'], # No output
                       batch_size=FFNN_hyper['batch_size'], # Number of observations per batch
                       validation_data=(X_test2, Y_test2)) # Data for evaluation
        y_pred_FFNN = FFNN_model.predict(X_test2) # Feedforward Neural Networks
        fAcc_test = skMetric.mean_squared_error( Y_test2, y_pred_FFNN)
        return fAcc_test
#------------------------------------------------------------------------------
#############################  PSO Algorithm ##################################
#------------------------------------------------------------------------------
# ---------------------- Rjb
# Parameter setting
PSO_hyper = {
        'Max_iter'    :  50, # number of neurons in a layer
        'Pop_size'    :  30, # activation function
        'dimensions'  :  2, # number of neurons in a layer
        'c1'          :  0.5, # activation function
        'c2'          :  0.3, # number of neurons in a layer
        'w'           :  0.9, # Mean squared error
        }
# Optimization of FFNN1 --------------------------------> best pos = [498.98, 519.548]
my_topology = Star() # The Topology Class
options = {'c1': PSO_hyper['c1'], 'c2': PSO_hyper['c2'], 'w':PSO_hyper['w']} # Set-up hyperparameters
bounds=[(X_train1.shape[0],X_train1.shape[0]), (2*X_train1.shape[0]+1,2*X_train1.shape[0]+1)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
optimizer = ps.single.GlobalBestPSO(n_particles=PSO_hyper['Pop_size'], dimensions=PSO_hyper['dimensions'], 
                                    options=options, bounds=bounds)
cost1, pos1 = optimizer.optimize(F1_1, iters=PSO_hyper['Max_iter'])
# Parameter setting
PSO_hyper = {
        'Max_iter'    :  50, # number of neurons in a layer
        'Pop_size'    :  30, # activation function
        'dimensions'  :  2, # number of neurons in a layer
        'c1'          :  0.5, # activation function
        'c2'          :  0.3, # number of neurons in a layer
        'w'           :  0.9, # Mean squared error
        }
# Optimization of FFNN2 --------------------------------> best pos = [9.229, 11.102]
my_topology = Star() # The Topology Class
options = {'c1': PSO_hyper['c1'], 'c2': PSO_hyper['c2'], 'w':PSO_hyper['w']} # Set-up hyperparameters
ub1=np.sqrt((1+2)*X_train1.shape[1])+2*np.sqrt(X_train1.shape[1]/(1+2))
ub2=(1+2)*np.sqrt(X_train1.shape[1]/(1+2))
bounds=[(X_train1.shape[1],X_train1.shape[1]), (ub1,ub2)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
optimizer = ps.single.GlobalBestPSO(n_particles=PSO_hyper['Pop_size'], dimensions=PSO_hyper['dimensions'], 
                                    options=options, bounds=bounds)
cost2, pos2 = optimizer.optimize(F2_1, iters=PSO_hyper['Max_iter'])
# ---------------------- Rrup
# Parameter setting
PSO_hyper = {
        'Max_iter'    :  50, # number of neurons in a layer
        'Pop_size'    :  30, # activation function
        'dimensions'  :  2, # number of neurons in a layer
        'c1'          :  0.5, # activation function
        'c2'          :  0.3, # number of neurons in a layer
        'w'           :  0.9, # Mean squared error
        }
# Optimization of FFNN1  --------------------------------> best pos = [676.74, 423.67]
my_topology = Star() # The Topology Class
options = {'c1': PSO_hyper['c1'], 'c2': PSO_hyper['c2'], 'w':PSO_hyper['w']} # Set-up hyperparameters
bounds=[(X_train2.shape[0],X_train2.shape[0]), (2*X_train2.shape[0]+1,2*X_train2.shape[0]+1)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
optimizer = ps.single.GlobalBestPSO(n_particles=PSO_hyper['Pop_size'], dimensions=PSO_hyper['dimensions'], 
                                    options=options, bounds=bounds)
cost3, pos3 = optimizer.optimize(F1_2, iters=PSO_hyper['Max_iter'])
# Parameter setting
PSO_hyper = {
        'Max_iter'    :  50, # number of neurons in a layer
        'Pop_size'    :  30, # activation function
        'dimensions'  :  2, # number of neurons in a layer
        'c1'          :  0.5, # activation function
        'c2'          :  0.3, # number of neurons in a layer
        'w'           :  0.9, # Mean squared error
        }
# Optimization of FFNN2  --------------------------------> best pos = [9.092, 9.520]
my_topology = Star() # The Topology Class
options = {'c1': PSO_hyper['c1'], 'c2': PSO_hyper['c2'], 'w':PSO_hyper['w']} # Set-up hyperparameters
ub1=np.sqrt((1+2)*X_train2.shape[1])+2*np.sqrt(X_train2.shape[1]/(1+2))
ub2=(1+2)*np.sqrt(X_train2.shape[1]/(1+2))
bounds=[(X_train2.shape[1],X_train2.shape[1]), (ub1,ub2)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
optimizer = ps.single.GlobalBestPSO(n_particles=PSO_hyper['Pop_size'], dimensions=PSO_hyper['dimensions'], 
                                    options=options, bounds=bounds)
cost4, pos4 = optimizer.optimize(F2_2, iters=PSO_hyper['Max_iter'])