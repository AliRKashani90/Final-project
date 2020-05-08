# -*- coding: utf-8 -*-
"""
Created on Mon May  4 21:41:21 2020

@author: alireza
"""
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_poisson_deviance
from sklearn.metrics import mean_gamma_deviance
from sklearn.metrics import mean_tweedie_deviance

#=======================================================================================
#                        prediction and accuracy
#=======================================================================================
#                      One-layer BPNN neural network
#------------------------------------------------------------------------------
############################  considering Rjb  ################################
#------------------------------------------------------------------------------
acc_FFNN_one_1Rjb = r2_score(Y_test1, y_pred_FFNN_one_1Rjb)
#acc_FFNN_one_2Rjb = r2_score(Y_test1, y_pred_FFNN_one_2Rjb)
#acc_FFNN_one_3Rjb = r2_score(Y_test1, y_pred_FFNN_one_3Rjb)
#acc_FFNN_one_4Rjb = r2_score(Y_test1, y_pred_FFNN_one_4Rjb)
#acc_FFNN_one_5Rjb = r2_score(Y_test1, y_pred_FFNN_one_5Rjb)
acc_FFNN_one_6Rjb = r2_score(Y_test1, y_pred_FFNN_one_6Rjb)
acc_FFNN_one_7Rjb = r2_score(Y_test1, y_pred_FFNN_one_7Rjb)
acc_FFNN_one_8Rjb = r2_score(Y_test1, y_pred_FFNN_one_8Rjb)
#acc_FFNN_one_9Rjb = r2_score(Y_test1, y_pred_FFNN_one_9Rjb)
#acc_FFNN_one_10Rjb = r2_score(Y_test1, y_pred_FFNN_one_10Rjb)
#acc_FFNN_one_11Rjb = r2_score(Y_test1, y_pred_FFNN_one_11Rjb)
#------------------------------------------------------------------------------
############################  considering Rrup  ###############################
#------------------------------------------------------------------------------
acc_FFNN_one_1Rrup = r2_score(Y_test2, y_pred_FFNN_one_1Rrup)
#acc_FFNN_one_2Rrup = r2_score(Y_test2, y_pred_FFNN_one_2Rrup)
#acc_FFNN_one_3Rrup = r2_score(Y_test2, y_pred_FFNN_one_3Rrup)
#acc_FFNN_one_4Rrup = r2_score(Y_test2, y_pred_FFNN_one_4Rrup)
#acc_FFNN_one_5Rrup = r2_score(Y_test2, y_pred_FFNN_one_5Rrup)
acc_FFNN_one_6Rrup = r2_score(Y_test2, y_pred_FFNN_one_6Rrup)
acc_FFNN_one_7Rrup = r2_score(Y_test2, y_pred_FFNN_one_7Rrup)
acc_FFNN_one_8Rrup = r2_score(Y_test2, y_pred_FFNN_one_8Rrup)
#acc_FFNN_one_9Rrup = r2_score(Y_test2, y_pred_FFNN_one_9Rrup)
#acc_FFNN_one_10Rrup = r2_score(Y_test2, y_pred_FFNN_one_10Rrup)
#acc_FFNN_one_11Rrup = r2_score(Y_test2, y_pred_FFNN_one_11Rrup)
###############################################################################
#                      Two-layer BPNN neural network
#------------------------------------------------------------------------------
############################  considering Rjb  ################################
#------------------------------------------------------------------------------
acc_FFNN_Two_1Rjb = r2_score(Y_test1, y_pred_FFNN_Two_1Rjb)
#acc_FFNN_Two_2Rjb = r2_score(Y_test1, y_pred_FFNN_Two_2Rjb)
#acc_FFNN_Two_3Rjb = r2_score(Y_test1, y_pred_FFNN_Two_3Rjb)
#acc_FFNN_Two_4Rjb = r2_score(Y_test1, y_pred_FFNN_Two_4Rjb)
#acc_FFNN_Two_5Rjb = r2_score(Y_test1, y_pred_FFNN_Two_5Rjb)
#acc_FFNN_Two_6Rjb = r2_score(Y_test1, y_pred_FFNN_Two_6Rjb)
#acc_FFNN_Two_7Rjb = r2_score(Y_test1, y_pred_FFNN_Two_7Rjb)
#acc_FFNN_Two_8Rjb = r2_score(Y_test1, y_pred_FFNN_Two_8Rjb)
#acc_FFNN_Two_9Rjb = r2_score(Y_test1, y_pred_FFNN_Two_9Rjb)
#acc_FFNN_Two_10Rjb = r2_score(Y_test1, y_pred_FFNN_Two_10Rjb)
#acc_FFNN_Two_11Rjb = r2_score(Y_test1, y_pred_FFNN_Two_11Rjb)
#------------------------------------------------------------------------------
############################  considering Rrup  ###############################
#------------------------------------------------------------------------------
acc_FFNN_Two_1Rrup = r2_score(Y_test2, y_pred_FFNN_Two_1Rrup)
#acc_FFNN_Two_2Rrup = r2_score(Y_test2, y_pred_FFNN_Two_2Rrup)
#acc_FFNN_Two_3Rrup = r2_score(Y_test2, y_pred_FFNN_Two_3Rrup)
#acc_FFNN_Two_4Rrup = r2_score(Y_test2, y_pred_FFNN_Two_4Rrup)
#acc_FFNN_Two_5Rrup = r2_score(Y_test2, y_pred_FFNN_Two_5Rrup)
#acc_FFNN_Two_6Rrup = r2_score(Y_test2, y_pred_FFNN_Two_6Rrup)
#acc_FFNN_Two_7Rrup = r2_score(Y_test2, y_pred_FFNN_Two_7Rrup)
acc_FFNN_Two_8Rrup = r2_score(Y_test2, y_pred_FFNN_Two_8Rrup)
#acc_FFNN_Two_9Rrup = r2_score(Y_test2, y_pred_FFNN_Two_9Rrup)
#acc_FFNN_Two_10Rrup = r2_score(Y_test2, y_pred_FFNN_Two_10Rrup)
#acc_FFNN_Two_11Rrup = r2_score(Y_test2, y_pred_FFNN_Two_11Rrup)
###############################################################################
#                      One-layer BPNN neural network
#------------------------------------------------------------------------------
############################  considering Rjb  ################################
#------------------------------------------------------------------------------
#OPT_acc_FFNN_one_1Rjb = r2_score(Y_test1, OPT_y_pred_FFNN_one_1Rjb)
OPT_acc_FFNN_one_2Rjb = r2_score(Y_test1, OPT_y_pred_FFNN_one_2Rjb)
OPT_acc_FFNN_one_3Rjb = r2_score(Y_test1, OPT_y_pred_FFNN_one_3Rjb)
OPT_acc_FFNN_one_4Rjb = r2_score(Y_test1, OPT_y_pred_FFNN_one_4Rjb)
OPT_acc_FFNN_one_5Rjb = r2_score(Y_test1, OPT_y_pred_FFNN_one_5Rjb)
OPT_acc_FFNN_one_6Rjb = r2_score(Y_test1, OPT_y_pred_FFNN_one_6Rjb)
OPT_acc_FFNN_one_7Rjb = r2_score(Y_test1, OPT_y_pred_FFNN_one_7Rjb)
#------------------------------------------------------------------------------
############################  considering Rrup  ###############################
#------------------------------------------------------------------------------
# One-layer BPNN neural network
#OPT_acc_FFNN_one_1Rrup = r2_score(Y_test2, OPT_y_pred_FFNN_one_1Rrup)
OPT_acc_FFNN_one_2Rrup = r2_score(Y_test2, OPT_y_pred_FFNN_one_2Rrup)
OPT_acc_FFNN_one_3Rrup = r2_score(Y_test2, OPT_y_pred_FFNN_one_3Rrup)
OPT_acc_FFNN_one_4Rrup = r2_score(Y_test2, OPT_y_pred_FFNN_one_4Rrup)
OPT_acc_FFNN_one_5Rrup = r2_score(Y_test2, OPT_y_pred_FFNN_one_5Rrup)
OPT_acc_FFNN_one_6Rrup = r2_score(Y_test2, OPT_y_pred_FFNN_one_6Rrup)
OPT_acc_FFNN_one_7Rrup = r2_score(Y_test2, OPT_y_pred_FFNN_one_7Rrup)
###############################################################################
###############################################################################
###############################################################################
#                      Two-layer BPNN neural network
#------------------------------------------------------------------------------
############################  considering Rjb  ################################
#------------------------------------------------------------------------------
#OPT_acc_FFNN_Two_1Rjb = r2_score(Y_test1, OPT_y_pred_FFNN_Two_1Rjb)
OPT_acc_FFNN_Two_2Rjb = r2_score(Y_test1, OPT_y_pred_FFNN_Two_2Rjb)
OPT_acc_FFNN_Two_3Rjb = r2_score(Y_test1, OPT_y_pred_FFNN_Two_3Rjb)
OPT_acc_FFNN_Two_4Rjb = r2_score(Y_test1, OPT_y_pred_FFNN_Two_4Rjb)
OPT_acc_FFNN_Two_5Rjb = r2_score(Y_test1, OPT_y_pred_FFNN_Two_5Rjb)
OPT_acc_FFNN_Two_6Rjb = r2_score(Y_test1, OPT_y_pred_FFNN_Two_6Rjb)
OPT_acc_FFNN_Two_7Rjb = r2_score(Y_test1, OPT_y_pred_FFNN_Two_7Rjb)
#------------------------------------------------------------------------------
############################  considering Rrup  ###############################
#------------------------------------------------------------------------------
#OPT_acc_FFNN_Two_1Rrup = r2_score(Y_test2, OPT_y_pred_FFNN_Two_1Rrup)
OPT_acc_FFNN_Two_2Rrup = r2_score(Y_test2, OPT_y_pred_FFNN_Two_2Rrup)
OPT_acc_FFNN_Two_3Rrup = r2_score(Y_test2, OPT_y_pred_FFNN_Two_3Rrup)
OPT_acc_FFNN_Two_4Rrup = r2_score(Y_test2, OPT_y_pred_FFNN_Two_4Rrup)
OPT_acc_FFNN_Two_5Rrup = r2_score(Y_test2, OPT_y_pred_FFNN_Two_5Rrup)
OPT_acc_FFNN_Two_6Rrup = r2_score(Y_test2, OPT_y_pred_FFNN_Two_6Rrup)
OPT_acc_FFNN_Two_7Rrup = r2_score(Y_test2, OPT_y_pred_FFNN_Two_7Rrup)