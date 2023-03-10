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
from scipy.signal import savgol_filter

## Order of all species
Species = ['Nitrate','Nitrite','Carbonate','Sulfate','Borate']

"""
Loading Data
"""
## These files should be in the same folder as your working directory or the path names should be given.
X = pd.read_csv('ATR-FTIR Spectra.csv', header=None).to_numpy()            #Spectra
y = pd.read_csv('ATR-FTIR Concentrations.csv', header=None).to_numpy()     #Concentrations
w = pd.read_csv('ATR-FTIR Wavenumbers.csv', header=None).to_numpy()        #ws

"""
Preprocessing
"""
## First Derivative Savitzky-Golay Filter with 5 filter points and a 2nd order polynomial
filter_points = 5
filter_order = 2
filter_deriv = 1
X = savgol_filter(X.copy(), filter_points,polyorder = filter_order,deriv=filter_deriv)

"""
Model
"""
## 15 latent variables chosen based on AIC
y_hat = np.zeros((np.shape(y))); y_hat_cls = y_hat.copy()
for i in range(np.size(y, axis = 0)):
    X_train = np.vstack((X[:i,:],X[(i+1):,:]))
    y_train = np.vstack((y[:i,:],y[(i+1):,:]))
    X_test = X[i:i+1,:]
    
    model = PLSRegression(n_components = 15, scale = False)
    model.fit(X_train,y_train)
    y_hat[i,:] = model.predict(X_test)
y_hat[y_hat<0]=0

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
color=['orangered','royalblue','limegreen','goldenrod','deeppink']
marker=['o','^','s','*','D','X']
fontsize=14
IRmin = 894
IRmax = 1798

plt.figure(dpi=300, figsize=(6,4.5))
plt.plot(w,X[:,:].T, color=color[4])
plt.xlim(IRmax,IRmin)
plt.title('ATR-FTIR Spectra (1st Derivative)')
plt.xlabel('Wavenumber ($\mathrm{cm^{-1}}$)', fontsize=fontsize)
plt.ylabel('Absorbance', fontsize=fontsize)

## Prediction parity plots
fontsize=14
maxplot = np.zeros((np.size(y, axis=1))); minplot = maxplot.copy()
for i in range(np.size(y[:,:], axis=1)):
    plt.figure(dpi=300, figsize=(6,4.5))
    x1=np.linspace(-100,100,101) 
    plt.plot(x1,x1,'k-') # identity line
    plt.plot(x1,.80*x1,'k--',alpha=.7)
    plt.plot(x1,1.2*x1,'k--',alpha=.7)
    plt.scatter(y[:,i],y_hat[:,i], color=color[i], marker=marker[4])
    plt.title(Species[i], color=color[i], fontsize=fontsize)
    plt.xlabel(r'Measured Molality $\mathrm{(\frac{mol}{kg \: solvent}}$)', fontsize=fontsize)
    plt.ylabel(r'Predicted Molality $\mathrm{(\frac{mol}{kg \: solvent}}$)', fontsize=fontsize)
    maxplot[i] = np.max(np.max((y[:,i],y_hat[:,i])))
    plt.xlim((0,maxplot[i]))
    plt.ylim((0,maxplot[i]))



