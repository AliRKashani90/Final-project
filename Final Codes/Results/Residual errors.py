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
#                       download and prep data
#=======================================================================================
# Split dataset to train and test sets.
dData=pd.read_csv('Test.csv', skiprows = []).as_matrix()
X = dData[:,:-1]
Y = dData[:,-1]

Predictions=pd.read_csv('Best.csv', skiprows = []).as_matrix()
Xp = Predictions[:,:-1]
Yp = Predictions[:,-1]
# ------------- Calculate Corr. matrix
title = ['T','Mw','Rjb','FRV','FNM','ZTOR','Delta','Vs_30','Z','Ln_Sa']
dData_dataframe = pd.DataFrame( dData, columns=title)
Predictions_dataframe = pd.DataFrame( Predictions, columns=title)
#=======================================================================================
#                       Plot the results
#=======================================================================================
"""
# ---------- Histogram
dData_dataframe.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
           xlabelsize=8, ylabelsize=8, grid=False)    
plt.tight_layout(rect=(0, 0, 1.2, 1.2))
# ---------- Pair-wise Scatter Plots
pp = sns.pairplot(dData_dataframe[title], size=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))

fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Spectral Acceleration Pairwise Plots', fontsize=14)
"""

"""
jp = sns.pairplot(dData_dataframe, x_vars=['Mw'], y_vars=['Ln_Sa'], size=4.5,
                  hue="wine_type", palette={"red": "#FF9999", "white": "#FFE888"},
                  plot_kws=dict(edgecolor="k", linewidth=0.5))


# ---------- Joint Scatter Plot
jp = sns.jointplot(x='sulphates', y='alcohol', data=wines,
                   kind='reg', space=0, size=5, ratio=4)

cols = ['density', 'residual sugar', 'total sulfur dioxide', 'fixed acidity', 'wine_type']
pp = sns.pairplot(wines[cols], hue='wine_type', size=1.8, aspect=1.8, 
                  palette={"red": "#FF9999", "white": "#FFE888"},
                  plot_kws=dict(edgecolor="black", linewidth=0.5))
fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Wine Attributes Pairwise Plots', fontsize=14)
"""
# ---------- Scatter Plot

title = ['T','Mw','Rjb','FRV','FNM','ZTOR','Delta','Vs_30','Z','Ln_Sa']

fig, ax = plt.subplots(nrows=3, ncols=3,sharex=True,sharey=True)

ax[0,0] = plt.subplot(331)
err1=dData_dataframe['Ln_Sa']-Predictions_dataframe['Ln_Sa']
plt.scatter(dData_dataframe['T'], err1,
            alpha=0.4, edgecolors='w')
plt.plot([np.min(dData_dataframe['T']),np.max(dData_dataframe['T'])],[0,0],'r--')
plt.xlabel('T (s)')

ax[0,1] = plt.subplot(332)
err1=dData_dataframe['Ln_Sa']-Predictions_dataframe['Ln_Sa']
plt.scatter(dData_dataframe['Mw'], err1,
            alpha=0.4, edgecolors='w')
plt.plot([np.min(dData_dataframe['Mw']),np.max(dData_dataframe['Mw'])],[0,0],'r--')
plt.xlabel('Mw')

ax[0,2] = plt.subplot(333)
err1=dData_dataframe['Ln_Sa']-Predictions_dataframe['Ln_Sa']
plt.scatter(dData_dataframe['Rjb'], err1,
            alpha=0.4, edgecolors='w')
plt.plot([np.min(dData_dataframe['Rjb']),np.max(dData_dataframe['Rjb'])],[0,0],'r--')
plt.xlabel('Rjb')

ax[1,0] = plt.subplot(334)
err1=dData_dataframe['Ln_Sa']-Predictions_dataframe['Ln_Sa']
plt.scatter(dData_dataframe['FRV'], err1,
            alpha=0.4, edgecolors='w')
plt.plot([np.min(dData_dataframe['FRV']),np.max(dData_dataframe['FRV'])],[0,0],'r--')
plt.xlabel('FRV')

ax[1,1] = plt.subplot(335)
err1=dData_dataframe['Ln_Sa']-Predictions_dataframe['Ln_Sa']
plt.scatter(dData_dataframe['FNM'], err1,
            alpha=0.4, edgecolors='w')
plt.plot([np.min(dData_dataframe['FNM']),np.max(dData_dataframe['FNM'])],[0,0],'r--')
plt.xlabel('FNM')

ax[1,2] = plt.subplot(336)
err1=dData_dataframe['Ln_Sa']-Predictions_dataframe['Ln_Sa']
plt.scatter(dData_dataframe['ZTOR'], err1,
            alpha=0.4, edgecolors='w')
plt.plot([np.min(dData_dataframe['ZTOR']),np.max(dData_dataframe['ZTOR'])],[0,0],'r--')
plt.xlabel('ZTOR')

ax[2,0] = plt.subplot(337)
err1=dData_dataframe['Ln_Sa']-Predictions_dataframe['Ln_Sa']
plt.scatter(dData_dataframe['Delta'], err1,
            alpha=0.4, edgecolors='w')
plt.plot([np.min(dData_dataframe['Delta']),np.max(dData_dataframe['Delta'])],[0,0],'r--')
plt.xlabel('Delta')

ax[2,1] = plt.subplot(338)
err1=dData_dataframe['Ln_Sa']-Predictions_dataframe['Ln_Sa']
plt.scatter(dData_dataframe['Vs_30'], err1,
            alpha=0.4, edgecolors='w')
plt.plot([np.min(dData_dataframe['Vs_30']),np.max(dData_dataframe['Vs_30'])],[0,0],'r--')
plt.xlabel('Vs_30')

ax[2,2] = plt.subplot(339)
err1=dData_dataframe['Ln_Sa']-Predictions_dataframe['Ln_Sa']
plt.scatter(dData_dataframe['Z'], err1,
            alpha=0.4, edgecolors='w')
plt.plot([np.min(dData_dataframe['Z']),np.max(dData_dataframe['Z'])],[0,0],'r--')
plt.xlabel('Z2.5')
#plt.xlabel('Magnitude')
#plt.ylabel('Ln(Sa)')
plt.suptitle('Residual Errors')
fig.text(0.04, 0.5, 'Error Values', va='center', rotation='vertical')
#plt.set_ylabel('Spectral Acc. (Ln(Sa))')
plt.show()
