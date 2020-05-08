# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:26:57 2020

@author: alireza
"""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression

from matplotlib import pyplot as plt
import seaborn as sns
import os
#=======================================================================================
#                       functions
#=======================================================================================
# Basic function to regenerate Ln(Sa)
def Ln_Sa_fun ( Coeff, Input):
    C_T = Coeff ['T']
    C_Mw = Coeff ['Mw']
    C_Rrup = Coeff ['Rrup']
    C_Rjb = Coeff ['Rjb']
    C_FRV = Coeff ['FRV']
    C_FNM = Coeff ['FNM']
    C_ZTOR = Coeff ['ZTOR']
    C_Delta = Coeff ['Delta']
    C_Vs_30 = Coeff ['Vs_30']
    C_Z = Coeff ['Z']
    C_Intercept = Coeff ['Intercept']
    return C_Intercept + C_T*Input[:,0] + C_Mw*Input[:,1] + C_Rrup*Input[:,2] + C_Rjb*Input[:,3] + C_FRV*Input[:,4] +\
    C_FNM*Input[:,5] + C_ZTOR*Input[:,6] + C_Delta*Input[:,7] + C_Vs_30*Input[:,8] + C_Z*Input[:,9]
#=======================================================================================
#                             parameters
#=======================================================================================
dir_out = r"C:\Users\alireza\Desktop\Project\1st (Lin_Reg)"
splitRatio = 1./4 # for training and test data
# The coefficients for the basic model based on multi-linear regression
Lin_reg_dummy_Coeff = {        
        'T'         : -0.4375, # ???
        'Mw'        :  0.6022, # ???
        'Rrup'      :  0.0623, # ???
        'Rjb'       : -0.0760, # ???
        'FRV'       :  0.3382, # ???
        'FNM'       : -0.3481, # ???
        'ZTOR'      : -0.0004, # ???
        'Delta'     :  0.0153, # ???
        'Vs_30'     : -0.0006, # ???
        'Z'         :  0.1249, # ???
        'Intercept' :  0.3921  # ???
        }
#=======================================================================================
#                       download and prep data
#=======================================================================================
# Split dataset to train and test sets.
dData=pd.read_csv('Data.csv', skiprows = []).as_matrix()
X = dData[:,:-1]
Y = dData[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split( dData[:,:-1], dData[:,-1],
                                    shuffle = True, test_size = splitRatio)
print('Train: Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
print('Test: Rows: %d, columns: %d'%(X_test.shape[0], X_test.shape[1]))
# ------------- Calculate Corr. matrix
title1 = ['T','Mw','Rrup','Rjb','FRV','FNM','ZTOR','Delta','Vs_30','Z']
X_dataframe = pd.DataFrame( X, columns=title1)
corrMatrix = X_dataframe.corr()
title2 = ['T','Mw','Rrup','Rjb','FRV','FNM','ZTOR','Delta','Vs_30','Z','Ln_Sa']
dData_dataframe = pd.DataFrame( dData, columns=title2)
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
#                         training and validation
#=======================================================================================
#------------------------------------------------------------------------------
############################  considering Rjb  ################################
#------------------------------------------------------------------------------
# ------------------- Linear regression without dummy
Lin_regressor_continuous1 = LinearRegression()
Lin_regressor_continuous1.fit(X_train1, Y_train1) #training the algorithm
# ------------------- Linear regression witho dummy
model1 = ols('Ln_Sa ~ T + Mw + Rjb + C(FRV) + C(FNM) + ZTOR + Delta + Vs_30 + Z', dData_dataframe1)
Lin_regressor_dummy1 = model1.fit ()
with open('summary_Rjb.txt', 'w') as fh:
    fh.write(Lin_regressor_dummy1.summary().as_text())
#------------------------------------------------------------------------------
###########################  considering Rrup  ################################
#------------------------------------------------------------------------------
# ------------------- Linear regression without dummy
Lin_regressor_continuous2 = LinearRegression()
Lin_regressor_continuous2.fit(X_train2, Y_train2) #training the algorithm
# ------------------- Linear regression witho dummy
model2 = ols('Ln_Sa ~ T + Mw + Rrup + C(FRV) + C(FNM) + ZTOR + Delta + Vs_30 + Z', dData_dataframe2)
Lin_regressor_dummy2 = model2.fit ()
with open('summary_Rrup.txt', 'w') as fh:
    fh.write(Lin_regressor_dummy2.summary().as_text())
#=======================================================================================
#                                 plots
#=======================================================================================
# --------- Plot Corr. Matrix
# plot correlation matrix
#fig1 = sns.heatmap(corrMatrix, annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', mask = np.tril(corrMatrix))
#file_out1 = 'Corr_matrix.png'
#os.chdir(dir_out)
#plt.savefig(file_out1)
# --------- Plot Corr. Matrix
# plot correlation matrix
#fig2 = sns.heatmap(corrMatrix1, annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', mask = np.tril(corrMatrix1))
#file_out2 = 'Corr_matrix1.png'
#os.chdir(dir_out)
#plt.savefig(file_out2)
# --------- Plot Corr. Matrix
# plot correlation matrix
#fig3 = sns.heatmap(corrMatrix2, annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', mask = np.tril(corrMatrix2))
#file_out3 = 'Corr_matrix2.png'
#os.chdir(dir_out)
#plt.savefig(file_out3)