# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:26:57 2020

@author: alireza
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import xlsxwriter
import pickle
#=======================================================================================
#                       download and prep data
#=======================================================================================
np.random.seed(123456)
dir_out = r"C:\Users\alireza\Desktop\Project"
splitRatio = 1./4 # for training and test data
#------------------------------------------------------------------------------
############################  considering Rjb  ################################
#------------------------------------------------------------------------------
# Split dataset to train and test sets.
dData1=pd.read_csv(r"C:\Users\alireza\Desktop\Project\SVR\Data1.csv", skiprows = []).as_matrix()
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
dData2=pd.read_csv(r"C:\Users\alireza\Desktop\Project\SVR\Data2.csv", skiprows = []).as_matrix()
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
# ------------------- Support vector regression (SVR)
# Linear model
svr_lin1 = SVR(kernel='linear', C=1, gamma='auto')
Lin_SVR_regressor1 = svr_lin1.fit(X_train1, Y_train1) #training the algorithm
# RBF model
svr_rbf1 = SVR(kernel='rbf', C=1, gamma=0.1, epsilon=.1)
RBF_SVR_regressor1 = svr_rbf1.fit(X_train1, Y_train1) #training the algorithm
# Polynomial model
svr_poly1 = SVR(kernel='poly', C=1, gamma='auto', degree=3, epsilon=.1, coef0=1)
Poly_SVR_regressor1 = svr_poly1.fit(X_train1, Y_train1) #training the algorithm
#------------------------------------------------------------------------------
###########################  considering Rrup  ################################
#------------------------------------------------------------------------------
# ------------------- Support vector regression (SVR)
# Linear model
svr_lin2 = SVR(kernel='linear', C=100, gamma='auto')
Lin_SVR_regressor2 = svr_lin2.fit(X_train2, Y_train2) #training the algorithm
# RBF model
svr_rbf2 = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
RBF_SVR_regressor2 = svr_rbf2.fit(X_train2, Y_train2) #training the algorithm
# Polynomial model
svr_poly2 = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)
Poly_SVR_regressor2 = svr_poly2.fit(X_train2, Y_train2) #training the algorithm
#=======================================================================================
#                        prediction and accuracy
#=======================================================================================
# Support vector regression
y_pred_Lin_SVR1 = Lin_SVR_regressor1.predict( X_test1) # Linear support vector regression
y_pred_RBF_SVR1 = RBF_SVR_regressor1.predict( X_test1) # RBF support vector regression
y_pred_Poly_SVR1 = Poly_SVR_regressor1.predict( X_test1) # Poly support vector regression
y_pred_Lin_SVR2 = Lin_SVR_regressor2.predict( X_test2) # Linear support vector regression
y_pred_RBF_SVR2 = RBF_SVR_regressor2.predict( X_test2) # RBF support vector regression
y_pred_Poly_SVR2 = Poly_SVR_regressor2.predict( X_test2) # Poly support vector regression
with open('y_pred_Lin_SVR1.pickle', 'wb') as f1:
    pickle.dump(y_pred_Lin_SVR1, f1)
with open('y_pred_RBF_SVR1.pickle', 'wb') as f2:
    pickle.dump(y_pred_RBF_SVR1, f2)

with open('y_pred_Poly_SVR1.pickle', 'wb') as g1:
    pickle.dump(y_pred_Poly_SVR1, g1)
with open('y_pred_Lin_SVR2.pickle', 'wb') as g2:
    pickle.dump(y_pred_Lin_SVR2, g2)
    
with open('y_pred_RBF_SVR2.pickle', 'wb') as g1:
    pickle.dump(y_pred_RBF_SVR2, g1)
with open('y_pred_Poly_SVR2.pickle', 'wb') as g2:
    pickle.dump(y_pred_Poly_SVR2, g2)
#---------------------------------------------------
workbook = xlsxwriter.Workbook('y_pred_Lin_SVR1.xlsx')
worksheet = workbook.add_worksheet()
row = 0
for col, data in enumerate(y_pred_Lin_SVR1):
    worksheet.write_column(row, col, data)
workbook.close()
#---------------------------------------------------
workbook = xlsxwriter.Workbook('y_pred_RBF_SVR1.xlsx')
worksheet = workbook.add_worksheet()
row = 0
for col, data in enumerate(y_pred_RBF_SVR1):
    worksheet.write_column(row, col, data)
workbook.close()
#---------------------------------------------------
workbook = xlsxwriter.Workbook('y_pred_Poly_SVR1.xlsx')
worksheet = workbook.add_worksheet()
row = 0
for col, data in enumerate(y_pred_Poly_SVR1):
    worksheet.write_column(row, col, data)
workbook.close()
#---------------------------------------------------
workbook = xlsxwriter.Workbook('y_pred_Lin_SVR2.xlsx')
worksheet = workbook.add_worksheet()
row = 0
for col, data in enumerate(y_pred_Lin_SVR2):
    worksheet.write_column(row, col, data)
workbook.close()
#---------------------------------------------------
workbook = xlsxwriter.Workbook('y_pred_RBF_SVR2.xlsx')
worksheet = workbook.add_worksheet()
row = 0
for col, data in enumerate(y_pred_RBF_SVR2):
    worksheet.write_column(row, col, data)
workbook.close()
#---------------------------------------------------
workbook = xlsxwriter.Workbook('y_pred_Poly_SVR2.xlsx')
worksheet = workbook.add_worksheet()
row = 0
for col, data in enumerate(y_pred_Poly_SVR2):
    worksheet.write_column(row, col, data)
workbook.close()