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
plt.scatter(dData_dataframe['T'], dData_dataframe['Ln_Sa'],
            alpha=0.4, edgecolors='w',label='plot 1')
plt.scatter(dData_dataframe['T'], Predictions_dataframe['Ln_Sa'],
            alpha=0.4, edgecolors='w')
plt.xlabel('T (s)')

ax[1,0] = plt.subplot(332)
plt.scatter(dData_dataframe['Mw'], dData_dataframe['Ln_Sa'],
            alpha=0.4, edgecolors='w')
plt.scatter(dData_dataframe['Mw'], Predictions_dataframe['Ln_Sa'],
            alpha=0.4, edgecolors='w')
plt.xlabel('Mw')

ax[2,0] = plt.subplot(333)
plt.scatter(dData_dataframe['Rjb'], dData_dataframe['Ln_Sa'],
            alpha=0.4, edgecolors='w')
plt.scatter(dData_dataframe['Rjb'], Predictions_dataframe['Ln_Sa'],
            alpha=0.4, edgecolors='w')
plt.xlabel('Rjb')

ax[1,0] = plt.subplot(334)
plt.scatter(dData_dataframe['FRV'], dData_dataframe['Ln_Sa'],
            alpha=0.4, edgecolors='w')
plt.scatter(dData_dataframe['FRV'], Predictions_dataframe['Ln_Sa'],
            alpha=0.4, edgecolors='w')
plt.xlabel('FRV')

ax[1,1] = plt.subplot(335)
plt.scatter(dData_dataframe['FNM'], dData_dataframe['Ln_Sa'],
            alpha=0.4, edgecolors='w')
plt.scatter(dData_dataframe['FNM'], Predictions_dataframe['Ln_Sa'],
            alpha=0.4, edgecolors='w')
plt.xlabel('FNM')

ax[1,2] = plt.subplot(336)
plt.scatter(dData_dataframe['ZTOR'], dData_dataframe['Ln_Sa'],
            alpha=0.4, edgecolors='w')
plt.scatter(dData_dataframe['ZTOR'], Predictions_dataframe['Ln_Sa'],
            alpha=0.4, edgecolors='w')
plt.xlabel('ZTOR')

ax[2,0] = plt.subplot(337)
plt.scatter(dData_dataframe['Delta'], dData_dataframe['Ln_Sa'],
            alpha=0.4, edgecolors='w')
plt.scatter(dData_dataframe['Delta'], Predictions_dataframe['Ln_Sa'],
            alpha=0.4, edgecolors='w')
plt.xlabel('Delta')

ax[2,1] = plt.subplot(338)
plt.scatter(dData_dataframe['Vs_30'], dData_dataframe['Ln_Sa'],
            alpha=0.4, edgecolors='w')
plt.scatter(dData_dataframe['Vs_30'], Predictions_dataframe['Ln_Sa'],
            alpha=0.4, edgecolors='w')
plt.xlabel('Vs_30')

ax[2,2] = plt.subplot(339)
plt.scatter(dData_dataframe['Z'], dData_dataframe['Ln_Sa'],
            alpha=0.4, edgecolors='w')
plt.scatter(dData_dataframe['Z'], Predictions_dataframe['Ln_Sa'],
            alpha=0.4, edgecolors='w')
plt.xlabel('Z2.5')
#plt.xlabel('Magnitude')
#plt.ylabel('Ln(Sa)')
plt.suptitle('Data Scatter for each Design variable')
fig.text(0.04, 0.5, 'Spectral Acc. (Ln(Sa))', va='center', rotation='vertical')
fig.legend(['Observations', 'Predictions'], loc='upper right')
#plt.set_ylabel('Spectral Acc. (Ln(Sa))')
plt.show()