# -*- coding: utf-8 -*-
"""
Published on: ...
@author: Steven Crouse
"""
import numpy as np
import pandas as pd
import scipy, glob, os, fnmatch, re, sklearn, time, sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cross_decomposition import PLSRegression


## Order of all species
Species = ['Kyanite','Wollastonite','Olivine','Silica','Zircon']

"""
Loading Data
"""
## These files should be in the same folder as your working directory or the path names should be given.
X = pd.read_csv('Raman Spectra.csv', header=None).to_numpy()            #Spectra
y = pd.read_csv('Raman Concentrations.csv', header=None).to_numpy()     #Concentrations
w = pd.read_csv('Raman Shifts.csv', header=None).to_numpy()             #Raman Shifts

"""
Model
"""

## Quantification model with leave-one-out cross validation
## 10 components for PLSR was chosen based on AIC minimum
y_hat = np.zeros((np.shape(y)))
for i in range(np.size(y, axis = 0)):
    X_train = np.vstack((X[:i,:],X[(i+1):,:]))
    y_train = np.vstack((y[:i,:],y[(i+1):,:]))
    X_test = X[i:i+1,:]
    
    model2 = PLSRegression(n_components = 10, scale = False)
    model2.fit(X_train, y_train)
    y_hat[i,:] = model2.predict(X_test)
y_hat[y_hat<0]=0

## Error Metrics
y_hat_resid = y_hat - y
y_hat_mean = np.mean(np.abs(y_hat_resid), axis=0)
y_hat_95 = np.percentile(np.abs(y_hat_resid), 95, axis=0)
y_hat_rmse = np.sqrt(np.sum(np.square(y_hat_resid), axis=0)/np.size(y_hat_resid, axis=0))
y_hat_R2 = sklearn.metrics.r2_score(y_hat, y, multioutput = 'raw_values')

y_hat_PE = np.zeros((len(Species)))
print('Percent Error for species:')
for i in range(len(Species)):
    y_hat_PE[i] = np.mean(np.abs(y_hat_resid[y[:,i]>0,i])/y[y[:,i]>0,i], axis=0)*100
    print(Species[i],': ', y_hat_PE[i])

"""
Plotting
"""

color=['orangered','royalblue','limegreen','goldenrod','darkviolet','slategray','chocolate','seagreen','dodgerblue','deeppink']
marker=['o','^','s','*','D','X']

## Plotting all Raman spectra
fontsize=14
plt.figure(dpi=300, figsize=(6,4.5))
plt.plot(w,X.T, color = color[1])
plt.xlim(100,1700)
plt.title('Raman Spectra')
plt.xlabel('Raman Shift ($\mathrm{cm^{-1}}$)', fontsize=fontsize)
plt.ylabel('Counts', fontsize=fontsize)

## Prediction Parity Plots
fontsize=14
maxplot = np.zeros((np.size(y, axis=1))); minplot = maxplot.copy()
for i in range(np.size(y[:,:], axis=1)):
    plt.figure(dpi=300, figsize=(6,4.5))
    x1=np.linspace(-100,400,101) 
    plt.plot(x1,x1,'k-') # identity line
    plt.plot(x1,.80*x1, 'k--',alpha=.7)
    plt.plot(x1,1.2*x1, 'k--',alpha=.7)
    plt.scatter(y[:,i],y_hat[:,i], color=color[4+i], marker=marker[4])
    plt.title(Species[i], color=color[4+i], fontsize=fontsize)
    plt.xlabel(r'Measured $\mathrm{(\frac{g\/\ solid}{kg\/\ solvent}}$)', fontsize=fontsize)
    plt.ylabel(r'Predicted $\mathrm{(\frac{g\/\ solid}{kg\/\ solvent}}$)', fontsize=fontsize)
    maxplot[i] = np.max(np.max((y[:,i],y_hat[:,i])))
    plt.xlim((0,maxplot[i]))
    plt.ylim((0,maxplot[i]))
    
    
